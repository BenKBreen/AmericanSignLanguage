### imports
import math
import numpy as np
import tensorflow as tf
import import_ipynb
from landmarks import *
from plotting import *
from fastai.vision.all import show_image
from collections import defaultdict


#################
### Functions ### 
#################

# Converts a sentence (string) into a list of characters without space 
def character_list(sentence):
    ans = []
    words = sentence.split()
    for word in words:
        prev = ''
        for i in word:
            if i in prev or not prev: 
                prev += i
            else:
                ans += [prev]
                prev = i       
        ans += [ prev ]
    return ans

### Encode / Decode

# Encode
E = {" ":0, "!":1, "#":2, "$":3, "%":4, "&":5, "'":6, "(":7, ")":8, "*":9, "+":10, ",":11, "-":12, ".":13, 
          "/":14, "0":15, "1":16, "2":17, "3":18, "4":19, "5":20, "6":21, "7":22, "8":23, "9":24, ":":25, ";":26,
          "=":27, "?":28, "@":29, "[":30, "_":31, "a":32, "b":33, "c":34, "d":35, "e":36, "f":37, "g":38, "h":39,
          "i":40, "j":41, "k":42, "l":43, "m":44, "n":45, "o":46, "p":47, "q":48, "r":49, "s":50, "t":51, "u":52, 
          "v":53, "w":54, "x":55, "y":56, "z":57, "~":58}
# Decode
D = {j:i for i,j in E.items()}

# Encode - converts letter to number
def Encode(x):
    return E[x]

# Decode - converts number to letter
def Decode(x):
    return D[x]



#######################
##### Video class #####
#######################
                
### Video Class
class video:           
    def __init__(self, datafile, seq_id):
        
        # initialize
        data = datafile.data
        data_label = datafile.data_labels.data
        
        self.id = seq_id 
        self.label = data_label.loc[ data_label['sequence_id'] == seq_id  ]
        self.phrase = self.label.phrase.iloc[0]
        self.participant = self.label.participant_id.iloc[0]
        self.data = data.loc[ seq_id ]
        self.frames = self.data.frame
        self.characters = character_list(self.phrase)
        self.nframes = 0 if type(self.frames) == np.float32 else len(self.frames)
    
    def __repr__(self): 
            a = " Sequence: {0}\n Phrase: {1}\n Signer: {2}\n Frames: {3}\n Dominant Hand: {4}\n Percentage: {5}".format( self.id, 
                                                                                                                          self.phrase,
                                                                                                                          self.participant, 
                                                                                                                          self.nframes, 
                                                                                                                          dominant_hand(self), 
                                                                                                                          percentage(self))

            return a 


        
        
#######################
### Video functions ###
#######################

def sequence(self):
    return self.id

def participant(self):
    return self.participant

def phrase(self):
    return self.phrase

def video_data(self):
    return self.data

def frames(self):
    return self.frames

def characters(self):
    return self.characters

def percentage(self):
    if type(self.frames) == np.float32: 
        return 0
    return round( 100 * len( hand_frames(self) ) / len( self.frames ), 2)
    

    
    
#######################
#### Dominant Hand ####
#######################

def dominant_hand(self):
    if hasattr(self, 'dominant_hand'):
        return self.dominant_hand
    
    # Initialize
    data = self.data
    
    # corner case
    if type(self.frames) == np.float32:
        
        # assign
        self.handframes = []
        self.dominant_hand = 'right'
        
        return self.dominant_hand
    
    # frames
    frames = data.frame.unique()
    
    # Left / right hand frames
    Lframes = [i for i in frames if not Landmark_is_nan(get_landmarks(self, 'left_hand',  i))]
    Rframes = [i for i in frames if not Landmark_is_nan(get_landmarks(self, 'right_hand', i))] 
    
    # Determine dominant hand 
    if len(Lframes) < len(Rframes):
        self.handframes = Rframes
        self.dominant_hand = 'right'
    else:
        self.handframes = Lframes
        self.dominant_hand = 'left'
    
    return self.dominant_hand

### Dominant hand frames
def hand_frames(self):
    dominant_hand(self) # this assigns handframes
    return self.handframes




#######################
### Normalized hand ###
#######################

def normal_hand_data(self):
    
    if hasattr(self, 'normal_hand_data'):
        return self.normal_hand_data, self.hand_position
    
    # hand frames 
    data = self.data
    Hframes = hand_frames(self)
    Dhand = dominant_hand(self)
    
    # loop 
    L, P = [], [] 
    for frame in Hframes:

        # Select landmarks for dominant hand
        X = get_landmarks(self, 'right_hand', frame) if Dhand == 'right' else get_landmarks(self, 'left_hand', frame)

        # First store position
        P.append( list(Landmark_vector(X,0)) )

        # center landmark
        X = center(X) if Dhand == 'right' else center(reflect(X))

        # List
        L.append( [ list(Landmark_vector(X,i)) for i in range(1,21) ] )

    # assign to video
    self.normal_hand_data, self.hand_position = L, P
    
    # return 
    return self.normal_hand_data, self.hand_position



#####################
####### Label #######
#####################

# TensorFlow Labels
# Takes as input a video V = self and model M with labeled clusters
def Label(self, M):
    
    # initialize 
    # characters = self.characters
    characters = [i[0] for i in self.characters] 
    frames, _ = normal_hand_data(self)
    n, m = len(characters), len(frames)
    
    # Need softmax later for predictions
    X = tf.keras.Sequential([ M, tf.keras.layers.Softmax() ])
    
    # Predictions
    Predictions = [] 
    for x in frames: 
        Predictions += [ X.predict(np.array([x]), verbose=0)[0] ]
    
    ## score function 
    # Given a letter (e.g. 'a') and cluster label c (e.g. 0) returns a score of the letter for that clump 
    def score( letter, i ):
        # print(letter,i)
        x = Encode( letter )
        s = Predictions[i][x]
        return s
    
    
    ### Prefix sums
    D = {}
    for c in characters:
        
        # initialize
        D[c] = defaultdict(int)
        pre, L = 0, []
        
        # loop
        for j in range(m):
            
            # update prefix sum
            pre += score(c,j)
            
            # loop over past prefix sum and compute score of i,j
            for i,x in enumerate(L):
                D[c][(i,j)] = (pre - x) / math.sqrt(j-i)
            
            # update list of prefixes
            L += [pre]
        
        # normalize scores
        maxim = max(D[c][y] for y in D[c])
        
        # normalize all scores
        for x in D[c]:
            D[c][x] /= maxim
            
    
    ### Dynamic programing
    
    # dp array
    dp = defaultdict( lambda : (0,[]) ) 
    for i, letter in enumerate(characters):
        
        # best + labels
        best, labs = 0, []
        
        # loop
        for j in range(m):
            
            # update best previous score
            if best < dp[(i-1,j-1)][0]:
                # print(best,labs)
                best, labs = dp[(i-1,j-1)]
            
            # loop over k
            for k in range(j+1,m):
                
                # length of interval
                l = k-j
                
                # average letter score for the frames j to k
                score_jk = best + D[letter][(j,k)]
                #print( (pre[letter][k] - pre[letter][j]) / l , letter, j,k)
                
                if dp[(i,k)][0] < score_jk:
                    
                    # update labels
                    labs_jk = labs + [ (j,k,letter) ]
                    
                    # update dp
                    dp[(i,k)] = ( score_jk, labs_jk ) 
    
    
    # optimal score
    maxim, labs = max([ dp[(n-1,j)] for j in range(m) ])

    # updating labels 
    newlabels = ['?'] * m
    for i, j, letter in labs:
        for k in range(i,j+1):
            newlabels[k] = letter
                
    # assign
    self.data_labels = newlabels
    
    # return labels
    return newlabels 



# return an individual label (once assigned)
def label(self, frame): 
    # check if labels are assigned
    if hasattr(self, 'data_labels'):
        return self.data_labels[frame]                                      
    return '?'


#####################
####### Label #######
#####################

### Predictions
def predict(self, M, frame):
            
    # data
    data, _ = normal_hand_data(self)
    data = np.array([ data[frame] ])
    
    # individual prediction
    X = tf.keras.Sequential([ M, tf.keras.layers.Softmax() ])
    p = X.predict(data, verbose=0)
    
    # prediction (encoded as an integer)
    pred = np.argmax(p)
    
    # answer
    ans = Decode(pred)
              
    # return
    return ans 




# Scrolling through image frames
# prediction_method = 'tflow' or 'kmeans'
def show_hands(V):    
    
    # initialize
    Hframes = hand_frames(V)
    frames = V.frames
    phrase = V.phrase
    Data = video_data(V)
    annotated_image = np.zeros((1024,1024,3),dtype=np.uint8)
    N, _ = normal_hand_data(V)
       
    # show function
    def show_frame(frame):
        # Only do frames with dominant hand
        hframe = Hframes[frame]

        # Get landmark data
        landmarks = get_landmarks_data(V, hframe)

        # Remove face and pose landmarks 
        landmarks.face_landmarks = Landmark_make_nan(landmarks.face_landmarks)
        landmarks.pose_landmarks = Landmark_make_nan(landmarks.pose_landmarks)

        # set dominant hand to be right
        if dominant_hand(V) == 'left': 
            landmarks.right_hand_landmarks = landmarks.left_hand_landmarks

        ### Plot original hand 
        landmarks.left_hand_landmarks = shift(landmarks.right_hand_landmarks, np.array([.2,.7,0]))
                
        ### Plot normalized hand
        Nhand = N[frame]
        
        # set right hand to normalized hand
        obj = landmark_pb2.LandmarkList()
        obj.landmark.add( x=0., y=0., z=0. ) 
        for v in Nhand:
            obj.landmark.add( x=v[0], y=v[1], z=v[2] ) 
        landmarks.right_hand_landmarks = shift( obj, np.array([.6,.7,0] ) )
        
        # show image 
        show_image(draw_landmarks(landmarks,annotated_image),figsize=(6,6),title=f'frame [{frame} of {len(frames)-1}] {phrase:50} Prediction: { label(V,frame) }')
        #show_image(draw_landmarks(landmarks,annotated_image),figsize=(6,6),title=f'frame [{frame+1} of {len(frames)}] {phrase:50} phrase: { V.phrase }')
        # print(f'showing frame: {frames[frame]}') # predict(V,frame)
    
    return show_frame

