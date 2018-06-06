#Just a test playground to mess around with bits and pieces before they're placed in the full-scale implementations.

import numpy as np;

weightMatrix = None
biasMatrix = None
inputMatrix = None

def relu_layer_forward(x):
    relu = lambda x: x * (x > 0).astype(float)
    return relu(x);
    
def read_weights_from_file(inputFile):
    global weightMatrix, biasMatrix
    with open(inputFile) as f:
        lines = f.readlines()
        numberOfLayers = int(lines[0])
        print numberOfLayers
        weightMatrix = np.empty(numberOfLayers, dtype=list)
        biasMatrix = np.empty(numberOfLayers, dtype=list)
        print weightMatrix
        currentLine = 2
        for i in range(numberOfLayers):
            dimensions = lines[currentLine].split(',')
            dimensions = [int(stringDimension) for stringDimension in dimensions]
            print dimensions
            print len(dimensions)
            currentLine += 1
            weights = [float(stringWeight) for stringWeight in lines[currentLine].split(',')]
            print len(weights)
            count = 0
            weightMatrix[i] = np.zeros((dimensions[0], dimensions[1]), dtype=float)
            for j in range(dimensions[1]):
                for k in range(dimensions[0]):
                    weightMatrix[i][k][j] = weights[count]
                    count += 1
            currentLine += 1
            biases = [float(stringBias) for stringBias in lines[currentLine].split(',')]
            biasMatrix[i] = np.zeros(len(biases))
            biasMatrix[i] = biases
            currentLine += 2
            
#Without a number in the format to specify the dimensions, I'm kind of at a loss here. I could assume they're square, which the particular examples I'm using here are, but that's not really generalizable. Also, pixels may be organized row- or column-wise, so that's also tricky.
def read_inputs_from_file(inputFile, height, width):
    global inputMatrix
    with open(inputFile) as f:
        lines = f.readlines()
        print len(lines), "examples"
        inputMatrix = np.empty(len(lines),dtype=list)
        for l in range(len(lines)):
            k = [float(stringIn) for stringIn in lines[l].split(',')[1:]] #This is to remove the useless 1 at the start of each string. Not sure why that's there.
            inputMatrix[l] = np.zeros((height, width),dtype=float)
            count = 0
            for i in range(height):
                for j in range(width):
                    inputMatrix[l][i][j] = k[count]
                    count += 1
            #inputMatrix[l] = np.transpose(k) #provides Nx1 output
        
            

#X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])

X = np.array([1, 3, -5, 6, 7])
a = (1, 2)
b = (0,)+ a
#input_padded = np.pad(X, (1,1), mode='constant')

#print X

#read_weights_from_file("./mnist_3A_layer.txt")
read_inputs_from_file("./example_10.txt", 28, 28)
print inputMatrix[0]