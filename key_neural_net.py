##########################################################################
## KEY DETECTION NEURAL NETWORK (PARALLEL DISTRIBUTED PROCESSING MODEL) ##
## Author: Grace Burnett                                                ##
##########################################################################

'''
ABOUT: There are 12 notes (C through B) and 12 major key signatures in Western music, with seven notes each.
A key signature can therefore be defined by the presence or absence of these 12 notes (represented by the matrix "keys" below).
This neural network represents the psychological process of musical key recognition.
'''


#import libraries
import numpy as np
import matplotlib.pyplot as plt


##  ============================ SET UP PHASE ============================= ##

# CREATION OF STIMULI

keys = np.array([
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],  # C
    [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],  # Db
    [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],  # D
    [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],  # Eb
    [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],  # E
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],  # F
    [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1],  # F#
    [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],  # G
    [1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],  # Ab
    [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],  # A
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],  # Bb
    [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1]   # B
])


# LAYER SET UP
# initialize layer sizes
sizeInput = 12
sizeHidden = 9
sizeOutput = 12

# initialize array to hold match values
matchTrace = []

# initialize arrays to represent dimensions of each layer
inputArr = np.empty(sizeInput)
hiddenArr = np.empty(sizeHidden)
outputArr = np.empty(sizeOutput)

# initialize arrays to represent input from each layer
hiddenNet = np.empty(sizeHidden)
outputNet = np.empty(sizeOutput)

# initialize momentum arrays
oldInp2Hid = np.zeros((sizeInput,sizeHidden))
oldHid2Out = np.zeros((sizeHidden,sizeOutput))


# GENERATE WEIGHTS
# btwn input and hidden
weightInp2Hid = np.random.uniform(low=-0.3, high=0.3, size=(sizeInput, sizeHidden))

# btwn hidden and output
weightHid2Out = np.random.uniform(low=-0.3, high=0.3, size=(sizeHidden, sizeOutput))


# INITIALIZE COUNT VARIABLES
epoch = 0
match = 0


# DEFINE ETA and MOMENTUM
eta = 0.4
momentum = 0.001



##  ============================ TRAINING PHASE ============================ ##

while match < 125 and epoch < 5000:
    epoch += 1
    match = 0
    teach = np.eye(12) # target output
    
    # iterate for each key
    for k in range(12):
        inputArr = keys[k]
        
        
        ## FEED FORWARD ##
        # determine net input and activation in hidden layer
        for i in range(sizeHidden):
            hiddenNet[i] = np.sum(weightInp2Hid[:, i] * inputArr)
            hiddenArr[i] = 1 / (1 + np.exp(-hiddenNet[i]))
        
        # determine net input and activation for output layer
        for i in range(sizeOutput):
            outputNet[i] = np.sum(weightHid2Out[:, i] * hiddenArr)
            outputArr[i] = 1 / (1 + np.exp(-outputNet[i]))
        
        # determine error in output layer
        deltaOutput = (teach[k] - outputArr) * outputArr * (1 - outputArr)
        
        
        ## BACK PROPOGATION ##
        # update weights between output and hidden layer
        for i in range(sizeOutput):
            for j in range(sizeHidden):
                weightHid2Out[j, i] += eta * deltaOutput[i] * hiddenArr[j] + momentum * oldHid2Out[j, i]
                oldHid2Out[j, i] = deltaOutput[i] * hiddenArr[j]
                
        # determine error in hidden layer
        deltaHidden = hiddenArr * (1 - hiddenArr) * np.dot(deltaOutput, weightHid2Out.T)
        
        # update weights between hidden and input layer
        for i in range(sizeHidden):
            for j in range(sizeInput):
                weightInp2Hid[j, i] += eta * deltaHidden[i] * inputArr[j] + momentum * oldInp2Hid[j, i]
                oldInp2Hid[j, i] = deltaHidden[i] * inputArr[j]
        
        
        ## UPDATE MATCH ##
        match += np.sum(((2 * outputArr) - 1) * ((2 * teach[k]) - 1))
    
    # print match and append to matchTrace
    print(round(match, 3), "out of 125 in epoch:", epoch)
    matchTrace.append(match)



##  ================================= PLOT ================================= ##

plt.plot(matchTrace)
plt.xlabel('epoch')
plt.ylabel('match')
plt.show()




