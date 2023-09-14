# imports 
import math
import numpy as np
import tensorflow as tf
from spellchecker import SpellChecker
from Levenshtein import distance
from collections import Counter
from collections import defaultdict

# Labeling

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

def find_phrase(frames):
    
    # initialize 
    m = len(frames)
    thresh = .1 #threshold
    n = 59
    
    # load model 
    M = tf.keras.models.load_model('asl/alphabetmodel2.h5')
    X = tf.keras.Sequential([ M, tf.keras.layers.Softmax() ]) # Need softmax later for predictions
    
    # Predictions
    Predictions = []
    for x in frames:
        if np.any(x):
            Predictions.append( X.predict(np.array(x), verbose=0)[0] )
        else:
            Predictions.append( np.array( [0.] * 59 )  )
    
    ## score function 
    # Given a letter (e.g. 'a') and cluster label c (e.g. 0) returns a score of the letter for that clump 
    def score( i, letter ):
        s = Predictions[i][letter]
        return s
    

    ### Prefix sums
    D = {}
    for c in range(59):
        
        # initialize
        D[c] = defaultdict(int)
        pre, L = 0, []
        
        # loop
        for j in range(m):
            
            # update prefix sum
            pre += score(j,c)
            
            # loop over past prefix sum and compute score of i,j
            for i,x in enumerate(L):
                D[c][(i,j)] = (pre - x) * (j-i) ** (1/3)
            
            # update list of prefixes
            L += [pre]
    
    ### Dynamic programming
    # dp array
    dp = defaultdict( lambda : (0,[]) )
    for i in range(m):
        
        if not np.any(Predictions[i]):
            continue
        
        # update past
        if dp[i][0] < dp[i-1][0]:
            dp[i] = dp[i-1]
        
        # update future
        for j in range(i+2, m):
            
            # loop
            for c in range(59):
                c_score = D[c][(i+1,j)]
                if dp[j][0] < c_score + dp[i][0]:
                     dp[j] = ( c_score + dp[i][0], dp[i][1] + [(i,j,c)] )
    
    # optimal score
    score, labs = dp[m-1]
    
    # new labels
    newlabels = ['?'] * m
    for i, j, letter in labs:
        for k in range(i,j+1):
            newlabels[k] = Decode(letter)
    
    # self.data_labels = newlabels
            
    # sentence
    s = ''.join( Decode(i[2]) for i in labs)
    
    # return s, labs, D, [ Decode[np.argmax(i)] for i in Predictions]
    return s, newlabels
    

# reconstruct sentence
def reconstruct_sentence(s):
    
    # initialize
    n = len(s)
    
    # spellchecker
    spell = SpellChecker()
    
    # dp array
    dp = defaultdict( lambda : (0,[]) )
    for i in range(n+1): dp[i] = (i, [])
    
    # loop
    for i in range(n):
        for j in range(i,n+1):
            
            # word
            word = s[i:j]
            
            # correct word
            cword = spell.correction(word)
            
            # distance
            d = len(word) if cword == None else distance(word, cword)
            
            # update dp array
            if dp[i][0] + d < dp[j][0]:
                dp[j] = ( dp[i][0] + d, dp[i][1] + [ cword ] )
             
    # score + words     
    score, words = dp[n]
    
    # sentence
    sen = ' '.join( words )
    
    # return sentence
    return sen






