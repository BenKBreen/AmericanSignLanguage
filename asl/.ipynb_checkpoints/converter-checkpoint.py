##### imports ######
import cv2
import numpy as np 
import pandas as pd 
import ipywidgets as widgets
import mediapipe as mp
import tensorflow as tf
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from fastai.vision.all import show_image
from ipywidgets import interact, interactive, fixed, interact_manual
from .landmarks import *
from .decode_phrase import *

# mediapipe functions
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


#############################
####### Hand detector #######
#############################

# (make sure hand_landmarker.task is in folder)
base_options = python.BaseOptions(model_asset_path='asl/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, min_hand_detection_confidence=.5, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)




#############################
###### Video detection ######
#############################

# use hand detector on video
def detect_landmarks(path):
    
    # asserts  
    vid = cv2.VideoCapture(path)
    assert vid.isOpened(), 'Make sure that the video is in a format accepted by c2v.VideoCapture()' 
    sucess, image = vid.read()
    assert sucess, 'Unable to load first frame of video'
    
    ### loop through frames and apply hand detetctor ### 
    count = 1
    images, landmarks = [], []
    while sucess:
        
        # import image to mediapipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # change colors from BGR to RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        
        # detect hand 
        result = detector.detect(mp_image)  
        
        # store
        images += [image ]
        landmarks += [ result ]
        
        # move to next frame
        sucess, image = vid.read()
        count += 1
        
    return images, landmarks 



# TODO: current only has right hand
def process_video(self):
    
    # rescale 
    m, n = self.resolution
    A = np.diag([ 950/m, 650/n , 1. ])
    

    L = []
    for frame in self.landmarks:
        if frame.hand_landmarks:
            
            # create the landmark
            x = landmark_pb2.LandmarkList()
            for v in frame.hand_landmarks[0]:
                v = np.matrix([v.x, v.y, v.z]) # make vector
                v = np.array(v @ A)[0] # scale
                x.landmark.add( x=v[0], y=v[1], z=v[2] ) 
            
            # center the landmark
            x = center(x)
            
            # extract the data 
            data = np.array([ [ Landmark_vector(x,i) for i in range(1,21) ] ])
            
            # add to L
            L += [ data ]
        else:
            L += [ np.array([ [ np.array([0.,0.,0.]) for i in range(20) ] ]) ]   
    return L


#######################
##### Video class #####
#######################
                
### Video file class
# set label = false if you do not want to automatically label the video
class video_file:           
    def __init__(self, path, label = True):
        
        # initialize
        frames, landmarkers = detect_landmarks(path)
        
        # attributes
        self.path = path
        self.frames = frames 
        self.landmarks = landmarkers
        self.total_frames = len(frames)
        self.resolution = 0 if self.total_frames == 0 else ( frames[0].shape[0], frames[0].shape[1] )
        self.total_hand_frames = len([ x for x in self.landmarks if x.hand_landmarks])
        self.landmark_percentage = 0 if self.total_frames == 0 else self.total_hand_frames / self.total_frames
        
        # label video
        if label:
            self.hand_frames = process_video(self)
            phrase, labels = find_phrase(self.hand_frames)
            self.phrase = phrase
            self.labels = labels
            

    def __repr__(self): 
            p = " path: {0}\n Resolution: {1}\n Number of frames: {2}\n Percentage of frames with landmarks: {3}".format( 
                self.path,
                self.resolution,                                                                                                      
                self.total_frames,                                                                                                  
                self.landmark_percentage ) 

            return p


        
        
        
############################
##### plotting a video #####
############################
       
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
        solutions.drawing_utils.draw_landmarks(
                                                annotated_image,
                                                hand_landmarks_proto,
                                                solutions.hands.HAND_CONNECTIONS,
                                                solutions.drawing_styles.get_default_hand_landmarks_style(),
                                                solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN
        
        # This is ushually wrong so we wont print
        # Draw handedness (left or right hand) on the image.
        #cv2.putText(annotated_image, f"{handedness[0].category_name}",
                #(text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                #FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image
  

# showing an annotated video frame function
def show_annotated_video(self):
    
    data = self.frames
    land = self.landmarks
    labels = self.labels
    phrase = self.phrase
    
    # show function
    def show_frame(i):
        
        # annotated image and show
        annotated_image = draw_landmarks_on_image(data[i], land[i])
        show_image(annotated_image, figsize=(6,6), title=f'Frame: {i} of {len(data)}   Phrase: {phrase}    Prediction: {labels[i]}')
        
    return show_frame


# showing an annotated video interact
def show_video(self):
    
    # check if assigned
    if hasattr(self, 'Video'):
        return self.Video
    
    # assign
    f = show_annotated_video(self)
    self.Video = interact(f, i=widgets.IntSlider(min=0, max=len(self.frames)-1, step=1, value=0, layout=widgets.Layout(width='1000px')))
    
    # return video
    return self.Video



# write video to a file
def write_video(self):
    
    # initialize
    data = self.frames
    land = self.landmarks
    labels = self.labels
    phrase = self.phrase
    hands = self.hand_frames
    
    # properties
    n_frames = len(data)
    height, width = self.resolution
    file_name = (self.path).split('/')[-1].split('.')[0] + '_annotated.avi'
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # final properties
    bord = 10
    fwidth = width + height + 4*bord
    fheight = height + 2*bord + 200
    
    # video file
    result = cv2.VideoWriter(file_name,  
                             cv2.VideoWriter_fourcc(*'MJPG'), 
                             10, 
                             (fwidth, fheight))
    
    # loop
    for i in range(n_frames):
        
        # get annotated image
        annotated_image = draw_landmarks_on_image(data[i], land[i])
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        annotated_image = cv2.copyMakeBorder(annotated_image,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255] )
        
        # draw normalized hand
        canvas = np.zeros((height,height,3),dtype=np.uint8)
        canvas[:] = (255, 255, 255)
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.add( x=0.5, y=0.6, z=0.0 ) # landmark 0
        hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x = 2*v[0]+0.5, y=2*v[1]+0.6, z=2*v[2]) for v in hands[i][0]]) # landmarks 1-20
        solutions.drawing_utils.draw_landmarks(
                                                canvas,
                                                hand_landmarks_proto,
                                                solutions.hands.HAND_CONNECTIONS,
                                                solutions.drawing_styles.get_default_hand_landmarks_style(),
                                                solutions.drawing_styles.get_default_hand_connections_style())
        canvas = cv2.copyMakeBorder(canvas,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255] )
        
        # add prediction
        title = f'Prediction: {labels[i]}'
        cv2.putText(canvas, title, (height // 4 + 100, height - 200), font, 2, (0,0,0), 3, 0)
        
        # put annotated image and canvas together
        annotated_image = cv2.hconcat((annotated_image, canvas))
        
        # header for text
        header = np.zeros((200, annotated_image.shape[1], 3), np.uint8)
        header[:] = (255, 255, 255) 
        annotated_image = cv2.vconcat((header, annotated_image))
        
        # add title
        title = f'Frame: {i} of {len(data)}   Phrase: {phrase}'
        cv2.putText( annotated_image, title, (fwidth // 3 + 200, 155), font, 2, (0,0,0), 3, 0)
        
        
        # write result
        result.write( annotated_image )
        
        # helpful prints for debugging
        # cv2.imshow('frame', annotated_image )
        #if cv2.waitKey(1) == ord('q'):
            #break

    result.release()
    cv2.destroyAllWindows()
    
    # print sucess
    print('Video saved')
    
    # return file name
    return file_name


