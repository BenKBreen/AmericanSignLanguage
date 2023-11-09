# American Sign Language

This package is video interpreter which can detect 59 distinct characters and symbols from American Sign Language. 

# Setup 
Complete the following steps to install the package. 

- Download the package.
- Install required python packages from the command line using

  ```pip install -r requirements.txt```
  
- Install the package from the command line using 

  ```pip install dist/American_Sign_Language_Reader-1.0-py3-none-any.whl```

# Usage
Select a video file that you would like to have interpreted. Navigate to the folder containing the file. Run the following command from the command line:

  ```aslread <video.file>```

# Method
Videos are processed in the following manner:

1. A video is first fed into the Mediapipe Hand Landmark Detector. The detector maps 21 distict landmarks points in the form of (x,y,z) coordinates onto each hand in a given frame. More information on the Mediapipe Hand Landmark detector can be found [here](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker). **Note:** The z-coordinate (depth) is often inaccurate.

2. For each frame, the 21 datapoints recovered from the Hand Landmark Detector are transfered to Tensorflow. Here the data is rescaled and centered so that the Hand Landmark datapoint 0 (cooresponding to a point on the wrist) is now at the origin. The hand is rotated into a fixed position which depends on whether the front or the back is facing the camera.

3. The normalized hand is then fed into Tensoflow CNN model. This model makes a prediction for each frame in the form of a probability distribution e.g. P(sign in frame i = 'u') = 0.5, P(sign in frame i = 'v') = 0.3, ...

4. Several scores are then created for the entire video based on character predictions for each frame (with a slight bonus for conseuctive frames predicting the same character). The best score is then converted into a phrase for the video.  

5. An annotated video is the created with the following information

-  Annotated original video with hand landmarks added
-  Normalized hand landmarks
-  Letter predictions
-  Phrase predictions

# Model 

The Tensorflow CNN model was trained on data from the Kaggle American Sign Language competition. Information can be found [here](https://www.kaggle.com/competitions/asl-fingerspelling). 




