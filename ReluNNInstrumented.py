#This is just an implementation of the Relu net in python. It can do either the small example or read from files, as you prefer. 

import numpy as np;
import matplotlib.pyplot as plt

n0 = 5
n1 = 3
n2 = 2

start = np.array([0.8, 0.4, 0.6, 0.3, 0.2])

#X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])

layer0 = np.zeros(n0, dtype=float)
layer1 = np.zeros(n1, dtype=float)
layer2 = np.zeros(n2, dtype=float)
sym_layer0 = np.zeros([n0, n0])
sym_layer1 = np.zeros([n1, n0])
sym_layer2 = np.zeros([n2, n0])
weights1 = np.empty([n0, n1])
weights2 = np.empty([n1, n2])
weightMatrix = None
biasMatrix = None
inputMatrix = None

b0 = 0.5
b1 = 0.5

#Manual initiation, for the SimpleNNInstrumented.java example Divya and Corina had
def init_simple():
    for i in range(n0):
        layer0[i] = start[i]
        sym_layer0[i, i] = 1
        print "sym_layer0 ",i
        print sym_layer0[i]
    
    for i in range(n0):
        for j in range (0, n1):
            weights1[i, j] = 0.5+i+j
        
    for i in range(n1):
        for j in range(0, n2):
            weights2[i, j] = 0.3+i
    
def init(inputFile, weightFile, inputHeight, inputWidth):
    read_inputs_from_file(inputFile, inputHeight, inputWidth)
    read_weights_from_file(weightFile)
    
def read_inputs_from_file(inputFile, height, width):
    global inputMatrix
    with open(inputFile) as f:
        lines = f.readlines()
        print len(lines), "examples"
        inputMatrix = np.empty(len(lines),dtype=list)
        for l in range(len(lines)):
            k = [float(stringIn) for stringIn in lines[l].split(',')[1:]] #This is to remove the useless 1 at the start of each string. Not sure why that's there, perhaps a "this-is-normalized-to" indicator?
            inputMatrix[l] = np.zeros((height, width),dtype=float) #Relu wants everything as row vectors, so height is almost always going to be 1. We could just assume that, but this is more generalizable. 
            count = 0
            for i in range(height):
                for j in range(width):
                    inputMatrix[l][i][j] = k[count] + 0.5
                    count += 1
                    
def read_weights_from_file(inputFile):
    global weightMatrix, biasMatrix
    with open(inputFile) as f:
        lines = f.readlines()
        numberOfLayers = int(lines[0])
        weightMatrix = np.empty(numberOfLayers, dtype=list)
        biasMatrix = np.empty(numberOfLayers, dtype=list)
        currentLine = 2
        for i in range(numberOfLayers):
            dimensions = lines[currentLine].split(',')
            dimensions = [int(stringDimension) for stringDimension in dimensions]
            #print dimensions
            currentLine += 1
            weights = [float(stringWeight) for stringWeight in lines[currentLine].split(',')]
            #print len(weights)
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
            
def relu_layer_forward(input, weights, b):
    out = np.add(np.dot(input, weights), b)
    print out
    relu = lambda x: x * (x > 0).astype(float)
    return relu(out);
    
#The transposes here are to make sure the output reads row-wise for the neurons in a layer, as opposed to row-wise for the pixels, i.e. out[i] is the coefficients for the ith neuron in terms of pixels 0...n-1. If you out[i] to show a pixel's impact on each neuron, remove the .transpose() calls. 
def sym_relu_layer_forward(input, weights):
    out = np.dot(input.transpose(), weights)
    out = out.transpose()
    relu = lambda x: x * (x > 0).astype(float)
    return relu(out);

def compute_layer1():
    global sym_layer1, layer1
    layer1 = relu_layer_forward(layer0, weights1, b0)
    sym_layer1 = sym_relu_layer_forward(sym_layer0, weights1)
    
    for i in range(sym_layer1.shape[0]):
        print "sym_layer1 ",i
        print sym_layer1[i]
    
def compute_layer2():
    global layer2, sym_layer2
    layer2 = relu_layer_forward(layer1, weights2, b1)
    sym_layer2 = sym_relu_layer_forward(sym_layer1, weights2)
    
    for i in range(sym_layer2.shape[0]):
        print "sym_layer2 ",i
        print sym_layer2[i]
    
def classify():
    maxValue=0
    maxIndex=-1
    for i in range(n2):
        print "Class",i," confidence:",layer2[i]
        if(layer2[i] > maxValue):
            maxValue = layer2[i]
            maxIndex = i
        
    print "MaxIndex:",maxIndex
    print sym_layer2[maxIndex]
    
def do_all_layers(inputNumber):
    global weightMatrix
    temp = inputMatrix[inputNumber]
    print temp.shape, weightMatrix.shape
    symTemp = np.identity(temp.shape[1])
    #print "SymTemp is an identity matrix of shape", symTemp.shape
    for i in range(len(weightMatrix)):
        temp = relu_layer_forward(temp, weightMatrix[i], biasMatrix[i])
        symTemp = sym_relu_layer_forward(symTemp, weightMatrix[i])
    maxValue=0
    maxIndex=-1
    for i in range(temp.shape[1]):
        print "Class",i,"confidence",temp[0][i]
        if(temp[0][i] > maxValue):
            maxValue = temp[0][i]
            maxIndex = i
        
    print "MaxIndex:",maxIndex
    print symTemp[maxIndex]
    plt.imshow(symTemp[maxIndex].reshape((28,28)))
    plt.show()

init("./example_10.txt", "./mnist_3A_layer.txt", 1, 784)
#for i in range(len(inputMatrix)):
do_all_layers(9)
    
#init_simple()
#print "layer0",layer0
#compute_layer1()
#print "layer1",layer1
#compute_layer2()
#print "layer2",layer2
#classify()