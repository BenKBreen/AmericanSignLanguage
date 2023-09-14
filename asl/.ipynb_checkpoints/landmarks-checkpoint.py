##############################
######### Landmarks ##########
##############################

### imports
import math
import pandas as pd
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import Counter
from mediapipe.framework.formats import landmark_pb2

# functions
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


### Initialize
number_of_landmarks = { 'face' : 468, 'left_hand' : 21, 'right_hand' : 21, 'pose' : 33 }

########################
#### Landmark class #### 
########################

### Landmark
class Landmarks(object):
    pass



##################
#### Creation #### 
##################

# Returns a "Landmark object" with a list of landmarks for the given object and frame   
def get_landmarks(self, name, frame):
    
    # asserts
    assert name in {'face', 'left_hand', 'right_hand', 'pose'}, 'name must be either ''face'', ''left_hand'', ''right_hand'', ''pose'''
    
    # initialize
    n = number_of_landmarks[name]
    data = self.data
    
    # loop
    obj = landmark_pb2.NormalizedLandmarkList()
    for i in range(n):
        
        # Column names in the data (dataframe) for the x,y,z coordinates of landmark i
        coordinates = {}
        for j in "xyz":
            column_name = j + '_' + name + '_' + str(i)
            coordinates[j] = data[ column_name ].loc[ data['frame'] == frame ]
        
        # coordinates
        xi,yi,zi = coordinates['x'], coordinates['y'], coordinates['z']
        
        # add to Landmark
        obj.landmark.add( x=xi, y=yi, z=zi )
    
    # return 
    return obj

# Creates all landmarks 
def get_landmarks_data(self, frame):    
    # set landmarks 
    result = Landmarks()
    result.face_landmarks = get_landmarks(self, 'face', frame)
    result.pose_landmarks = get_landmarks(self, 'pose', frame)
    result.left_hand_landmarks = get_landmarks(self, 'left_hand', frame)
    result.right_hand_landmarks = get_landmarks(self, 'right_hand', frame)
    return result





##################
### Operations ### 
##################

# Checks if a landmark is not defined
def Landmark_is_nan(X):
    for i in X.landmark:
        if math.isnan(i.x) or math.isnan(i.y) or math.isnan(i.z):
            return True
    return False


# returns <x,y,z> vector for landmark n
def Landmark_vector(X,i):
    return np.array([ X.landmark[i].x,  X.landmark[i].y,  X.landmark[i].z])
 
    
# Makes a blank landmark of length n
def Landmark_blank(n):
    obj = landmark_pb2.NormalizedLandmarkList()
    for i in range(n):
        obj.landmark.add( x=math.nan, y=math.nan, z=math.nan )
    return obj


# Creates a blank for all landmarks
def blank():  
    result = Landmarks()
    result.face_landmarks = Landmark_blank(468)
    result.pose_landmarks = Landmark_blank(33)
    result.left_hand_landmarks = Landmark_blank(21)
    result.right_hand_landmarks = Landmark_blank(21)
    return result


# Makes a landmark undefined
def Landmark_make_nan(X):
    obj = landmark_pb2.NormalizedLandmarkList()
    for i in X.landmark:
        obj.landmark.add( x=math.nan, y=math.nan, z=math.nan )
    return obj


# Shifts a landmark to be centered at the vector w
def shift(X,w):
    obj = landmark_pb2.NormalizedLandmarkList()
    a = np.array([ X.landmark[0].x , X.landmark[0].y, X.landmark[0].z ]) 
    for i in X.landmark:
        # form vector
        v = np.array([i.x,i.y,i.z])
        # unshift affine shift
        v = v - a + w
        # set coordinates
        obj.landmark.add( x=v[0], y=v[1], z=v[2] )       
    return obj 


# reflects a landmark about the y-axis
def reflect(X):
    obj = landmark_pb2.LandmarkList()
    for i in X.landmark:
        obj.landmark.add( x=(.5-i.x), y=i.y, z=i.z )       
    return obj 


# records whether front or back of hand
def front(X):
    
    # points
    X0 = Landmark_vector(X,0) # base
    X5 = Landmark_vector(X,5) # left 
    X17 = Landmark_vector(X,17) # right
    
    # vectors 
    V1, V2 = X5 - X0, X17 - X0
    
    # set z to 0 
    # I think this is stupid - third entry of cross product is independent from these values
    V1[2], V2[2] = 0,0
    
    # take cross product
    n = np.cross(V1,V2)
    
    return True if n[2] < 0 else False  



# Recenters a hand based on the outer landmarks 
def center(X):
    
    # if not defined
    if Landmark_is_nan(X): return X

    # Centering
    X0 = Landmark_vector(X,0) # base
    X5 = Landmark_vector(X,5) # center left 
    X17 = Landmark_vector(X,17) # right
        
    # Basis 1
    V5, V17 = X5-X0, X17-X0
    normal = np.cross(V5, V17)

    # Basis Matrices
    A, B = np.array([normal,V5,V17]),    np.array([[ 0.00605257,  0.00246374, -0.01169141], 
                                                   [ 0.04396006, -0.13104451, -0.0048572 ], 
                                                   [-0.05505851, -0.10182643, -0.04996139]])
    A1, B1 = np.array([normal,V5,V17]),  np.array([[ 0.01846211, -0.00295458,  0.02261392], 
                                                   [ 0.12170109, -0.12008113, -0.11504634], 
                                                   [ 0.18375406,  0.00450712, -0.14942883]])
    
    # Transformation
    C = (np.linalg.inv(A)).dot(B)
    if not front(X): 
        C = (np.linalg.inv(A1)).dot(B1)
        # print('back')
    
    # SVD
    U, S, V = np.linalg.svd(C, full_matrices=True)
    C = U.dot(V)
    
    # resize
    scale = .01
    mag = np.linalg.norm( np.cross(V5,V17), 2)
    rvalue = ( scale / mag) ** (1./3) # print(rvalue)
    D = np.diag(np.full(3,rvalue))
    C = D.dot(C)
      
    obj = landmark_pb2.LandmarkList()
    for i in X.landmark:
            
            # form vector
            v = np.array([i.x,i.y,i.z])
            
            # affine shift
            v = v - X0
            
            # linear transformation
            v = v.dot(C)
            
            # set coordinates
            obj.landmark.add( x=v[0], y=v[1], z=v[2] )
            
    return obj 



##################
### Plotting ### 
##################

def draw_landmarks(landmarks,image,show_pose=True,show_face_contour=True,show_face_tesselation=True,show_left_hand=True,show_right_hand=True):
    annotated_image = image.copy()
    results = landmarks
    if show_face_tesselation:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
    if show_face_contour:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
    if show_pose:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.
            get_default_pose_landmarks_style())
    if show_left_hand:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_hand_landmarks_style())
    if show_right_hand:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_hand_landmarks_style())
    return annotated_image
