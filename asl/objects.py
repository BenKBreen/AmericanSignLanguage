##############################
######## Data Labels #########
##############################

### imports
import pandas as pd
from collections import Counter



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





##################
### Data Class ### 
##################

### Data Labels 
''' 
This imports the metadata for all datafiles in either the 
supplemental dataset or the main dataset.
''' 
class metadata:           
    def __init__(self, ID):
        
        # assert 
        assert ID in { 'supplemental' , 'train' }, 'Make sure that ID is either ""supplemental"" or ""train""' 
        
        # select path
        if ID == 'supplemental': path = '/Volumes/My Passport for Mac/asl-fingerspelling/supplemental_metadata.csv' 
        elif ID == 'train': path = '/Volumes/My Passport for Mac/asl-fingerspelling/train.csv' 
        
        # load dataframe
        data = pd.read_csv(path)
        
        # attributes
        self.id = ID
        self.data = data
        self.files = data.file_id.unique()
        self.participants = data.participant_id.unique()
        self.phrases = data.phrase.unique()
        # self.phrases_counts = { i : data.phrase.unique() } should count number of element
        self.phrases_by_characters = [ character_list(i) for i in self.phrases ] # Convert each phrase into list of characters
        self.character_counts = Counter( sum(self.phrases_by_characters, []) ) # Count occurance of characters across all sentences
        self.characters = sorted(list(self.character_counts))

        # Encoding and decoding characters as integers
        Encode, Decode = {}, {}
        for i,x in enumerate(self.characters):
            Encode[x] = i
            Decode[i] = x
        
        self.encode, self.decode = Encode, Decode
        
             
### Data File 
'''
Loads a single datafile which contains ~1000 videos + data
Inputs: datalabel (created above) and ID from 
'''
class datafile:           
    def __init__(self, datalabel, file):
        
        # load dataframe 
        if datalabel.id == 'supplemental':
            path = '/Volumes/My Passport for Mac/asl-fingerspelling/supplemental_landmarks/' + file + '.parquet'
        else:
            path = '/Volumes/My Passport for Mac/asl-fingerspelling/train_landmarks/' + file + '.parquet'
        
        # data
        data = pd.read_parquet(path)
        
        # attributes
        self.id = file
        self.data = data
        self.data_labels = datalabel
        self.videos = data.index.unique()
        
        
        


