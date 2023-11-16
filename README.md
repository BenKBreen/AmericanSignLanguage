# American Sign Language

This package is video interpreter which can detect 59 distinct characters and symbols from American Sign Language. 

Ex: An attempt to sign "Hi Artem"

<img src="ASL.gif" width="900" height="450"/>


# Setup 
Complete the following steps to install the package. 

- Download the package.
- Install required python packages from the command line using:

  ```pip install -r requirements.txt```
  
- Install the package from the command line using: 

  ```pip install dist/American_Sign_Language_Reader-1.0-py3-none-any.whl```

# Usage
Select a video file that you would like to have interpreted. Navigate to the folder containing the file. Run the following command from the command line:

  ```aslread <path to video>```

# Method
Videos are processed in the following manner:

1. A video is first fed into the Mediapipe Hand Landmark Detector. The detector maps 21 distict landmarks points in the form of (x,y,z) coordinates onto each hand in a given frame. More information on the Mediapipe Hand Landmark detector can be found [here](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker).
  
   **Note:** The z-coordinate (depth) is often inaccurate.

2. The landmark data is transfered into Tensorflow. The datapoints are rescaled, centered so that Hand Landmark datapoint 0 (datapoint on the wrist) is now at the origin, and rotated into a fixed position using an orthogonal transformation. 

3. The processed datapoints are fed into a Tensoflow CNN model trained on data from the American Sign Language competition on Kaggle. The model makes frame-by-frame predictions in the form of probability distributions representing the likelihood of each of the 59 characters to represent the sign in a given frame e.g. P(sign in frame i = 'u') = 0.5, P(sign in frame i = 'v') = 0.3, ...

4. A phrase is assigned to the entire video. This is done in the folling manner: we consider all labelings of the frames with the 59 distinct characters and assign a score to each labeling based on the aggragate sum of the model predicting the given character for a given frame with a slight weighting bonus for consecutive frames predicting the same character. The best score is converted to a phrase for the video.  

5. An annotated video is the created with the following information:

    -  Annotated original video with hand landmarks added
    -  Normalized hand landmarks
    -  Letter predictions
    -  Phrase predictions

# Model 

The Tensorflow CNN model was trained on data from the Kaggle American Sign Language competition. Information can be found [here](https://www.kaggle.com/competitions/asl-fingerspelling). 

# Acknowledgements

I would like to thank my mentor Artem Yankov for guiding me through this project. 




