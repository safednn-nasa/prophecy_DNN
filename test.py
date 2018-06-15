#Just a test playground to mess around with bits and pieces before they're placed in the full-scale implementations.

import numpy as np;
import matplotlib.pyplot as plt
import time

weightMatrix = None
biasMatrix = None
inputMatrix = None

#Assumed format of inputs file: list of inputs of size S+1, where the first element is discardable.
#Populates inputMatrix, a matrix of N height-by-width-by-1 3D matrices, where each is an input. To make it N Sx1x1 column vecotrs, replace final nested for loops with final line. For N 1xSx1 row vectors, replace np.transpose(k) with k.
def read_inputs_from_file(inputFile, height, width):
    global inputMatrix
    with open(inputFile) as f:
        lines = f.readlines()
        print len(lines), "examples"
        inputMatrix = np.empty(len(lines),dtype=list)
        for l in range(len(lines)):
            k = [float(stringIn) for stringIn in lines[l].split(',')[1:]] #This is to remove the useless 1 at the start of each string. Not sure why that's there.
            inputMatrix[l] = np.zeros((1, 1, height, width),dtype=float) #we're asuming that everything is one 2D image for now. The 1s are just to keep numpy happy.
            count = 0
            for i in range(height):
                for j in range(width):
                    inputMatrix[l][0][0][i][j] = k[count] +0.5
                    count += 1
            #inputMatrix[l] = np.transpose(k) #provides Nx1 output
    
#Assumed format of weights file: [N=numberOfLayers\n\n(X=heightOfFilter,Y=widthOfFilter\n(Z=csvOfXTimesYWeights)\n(B=csvOfYBiases)\n\n)*N]
#Populates weightMatrix, a 3D matrix of N 2D matrices with X rows and Y columns, and biasMatrix, a 2D matrix of N rows where each row is a list of Y biases for the respective filters 
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

def get_im2col_indices(input_shape, field_height, field_width, padding=1, stride=1):
    N, C, H, W = input_shape
    print "Im2Col input shape:", input_shape
    #C, H, W = input_shape
    #print "C, H, W: ",C,H,W
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1
    
    print "Out height, width:", out_height, out_width
    
    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    #print i0
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    
    return (k, i, j)
     
def im2col_indices(input, section_height, section_width, padding=1, stride=1):
    input_padded = np.pad(input, ((0,0),(0,0),(padding, padding),(padding, padding)), mode='constant')
    k, i, j = get_im2col_indices(input.shape, section_height, section_width, padding, stride)
    #print k, i, j
    cols = input_padded[:, k, i, j]
    #print cols
    C = input.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(section_height * section_width * C, -1)
    return cols

def im2col_sliding_strided(A, block_shape, stepsize=1, padding=0):
    d,m,n = A.shape
    A_padded = np.pad(A, ((0,0),(padding, padding),(padding, padding)), mode='constant')
    #print "Im2col input shape:",m, n
    s2, s0, s1 = A_padded.strides   
    #print "Im2col input strides",s0, s1 
    nrows = m-block_shape[0]+1
    ncols = n-block_shape[1]+1
    #print "Im2col output # of rows, cols:",nrows, ncols
    shp = block_shape[0],block_shape[1],nrows,ncols
    print shp
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(A_padded, shape=shp, strides=strd)
    return out_view.reshape(block_shape[0]*block_shape[1],-1)[:,::stepsize]

def conv_layer_forward(input, filter, b, stride=1, padding=1):
    print "Beginning convolutional layer"
    #print "Conv: input shape",input.shape
    n_x, d_x, h_x, w_x = input.shape #Still assuming one 2D image at a time for now, so d_x should start as 1 and then depend on the number of filters, n_x is always 1
    #print "Conv: filter shape",filter.shape
    n_filters, d_filter, h_filter, w_filter = filter.shape 
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1
    input_col = im2col_indices(input, h_filter, w_filter, padding=padding, stride=stride) #im2col_sliding_strided(input, (h_filter, w_filter), stride, padding) 
    print "Input_col shape",input_col.shape
    filter_col = filter.reshape(n_filters, -1)
    print "Filter_col shape",filter_col.shape
    converted_biases = np.array(b).reshape(-1, 1)
    #print "10x1?",converted_biases.shape
    out = np.add(np.dot(filter_col, input_col), converted_biases)
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2) #Turns it back to (n_x, n_filters, h_out, w_out), where n_x should be 1
    #print "output shape",out.shape
    print ""
    return out;
    
#TODO: Finish this for-loop version for individual images. No n_x here.
def conv_layer_forward_ineff(input, filters, biases, stride=1, padding=1):
    print "Beginning conv layer"
    print input.shape
    n_x, d_x, h_x, w_x = input.shape 
    n_filters, d_filter, h_filter, w_filter = filters.shape #d_x should equal d_filter
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1
    print h_out, "x", w_out, "out"
    input_padded = np.pad(input,((0,0),(0,0),(padding, padding),(padding, padding)),mode='constant')
    out = np.zeros((n_x, n_filters, h_out, w_out))
    for im in range(n_x):
        for i in range(n_filters):
            for j in range(0, h_out, stride):
                for k in range(0, w_out, stride):
                    for l in range(d_x):
                        #do a dot product between filter and current piece of padded input
                        #print filters[i][l].shape
                        #print input_padded[im,l,j:j+h_filter,k:k+w_filter].shape
                        dot_out = np.multiply(filters[i][l], input_padded[im,l,j:j+h_filter,k:k+w_filter])
                        #print np.sum(dot_out)
                        out[im][i][j][k] = out[im][i][j][k] + np.sum(dot_out)
                    out[im][i][j][k] = out[im][i][j][k] + biases[i]
                    print "Total for this filter:", out[im][i][j][k]
    print ""
    return out;
    
def relu_layer_forward(x):
    relu = lambda x: x * (x > 0).astype(float)
    return relu(x);
    
#We could make this multi-channel with an "n" parameter, changes needed noted in comments
def pool_layer_forward(X, size, stride = 1):
    print "Beginning pooling layer"
    print "X shape:",X.shape
    n, d, h, w = X.shape
    h_out = h/size
    w_out = w/size
    X_reshaped = X.reshape(n*d, 1, h, w)
    X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride) #im2col_sliding_strided(X_reshaped, (size, size), stepsize=stride) #
    #print "X_col shape:",X_col.shape
    max_idx = np.argmax(X_col, axis=0)
    out = X_col[max_idx, range(max_idx.size)]
    out = out.reshape(h_out, w_out, n, d)
    out = out.transpose(2, 3, 0, 1)
    print ""
    return out;
    
def pool_layer_forward_ineff(X, size, stride = 1):
    print "X shape:",X.shape
    n, d, h, w = X.shape
    h_out = (h-size)/stride + 1
    w_out = (w-size)/stride + 1
    out = np.empty((n, d, h_out, w_out))
    for im in range(n):
        for i in range(0, h_out, stride):
            for j in range(0, w_out, stride):
                for k in range(d):
                    #print i, j, k
                    #print X[im][k][i:i+size][j:j+size]
                    #print X[im][k][i:i+size][j:j+size].max()
                    out[im][k][i][j] = X[im][k][i:i+size][j:j+size].max()
    return out

#Each node in the layer must have an array with a list input.size of coefficients. Aside from the FC->Convolution conversion case, can we even do that as a matrix operation? For FC layer, it's just a dot product: previous sym_layer with the weight matrix. 
#If our input is X x Y, every point on a given internal/output layer needs an X x Y map of coefficients. Can we create a weight matrix by overlapping the filters by their stride? That would be the total impact a pixel had overall on the next layer, but not appropriately divided.
def sym_conv_layer_forward(input, filter, b, stride=1, padding=1):
    #TODO: Figuring out how to implement this is the core of the project.
    #out = np.dot(input, filter)
    #out = out.transpose()
    #print "Conv: input shape",input.shape
    h_x, w_x, d_x = input.shape #Still assuming one 2D image at a time for now, so we omit the n_x and d_x should always be 1
    #print "Conv: filter shape",filter.shape
    n_filters, h_filter, w_filter = filter.shape
    #Want to produce an output of shape (n_filters, h_x, w_x). Using stride=1 and padding=(filter_size-1)/2 will at least guarantee that h_out = h_x and w_out = w_x
    padding = (h_filter-1)/2
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1
    input_col = im2col_sliding_strided(input, (h_filter, w_filter), stride, padding) #im2col(input, filter.shape[0], filter.shape[1], padding=padding, stride=stride)
    #print "Input_col shape",input_col.shape
    filter_col = filter.reshape(n_filters, -1)
    #print "Filter_col shape",filter_col.shape
    #converted_biases = np.array(b).reshape(-1, 1)
    #out = np.add(np.dot(filter_col, input_col), converted_biases)
    out = out.reshape(n_filters, h_out, w_out) #(n_filters, h_out, w_out, n_x) if multi-input
    #out = out.transpose(3, 0, 1, 2) #Turns it back to (n_x, n_filters, h_out, w_out), but we don't need that since we don't have multiple inputs
    #print "output shape",out.shape
    return out;

def init(inputFile, weightFile, inputHeight, inputWidth):
    read_inputs_from_file(inputFile, inputHeight, inputWidth)
    read_weights_from_file(weightFile)
    
def classify(processedArray):
    maxValue = 0
    maxIndex = -1
    for i in range(processedArray.shape[1]):
        print "Class",i,"confidence",processedArray[0][i][0][0]
        if(processedArray[0][i][0][0] > maxValue):
            maxValue = processedArray[0][i][0][0]
            maxIndex = i
    print "MaxIndex:",maxIndex
    
def reshape_fc_weight_matrix(fcWeights, proper_shape):
    total_height, n_filters = fcWeights.shape
    proper_depth, proper_height, proper_width = proper_shape
    temp = np.empty((n_filters, proper_depth, proper_height, proper_width))
    #Each column of an FC weight matrix is a filter that will be placed over the entire input. We want to turn each of them into a proper_height x proper_width matrix, and end up with something of shape (n_filters, proper_depth, proper_height, proper_width). 
    for i in range(n_filters):
        for j in range(proper_depth):
            for k in range(proper_height):
                for l in range(proper_width):
                    index = k*proper_width + l
                    temp[i][j][k][l] = fcWeights[index][i]
    return temp
    
def extents(f):
  delta = f[1] - f[0]
  return [f[0] - delta/2, f[-1] + delta/2]

#OKAY SO
#The input is of shape (number_of_images, depth_of_images, height_of_images, width_of_images)
#The weights are of shape (number_of_filters, depth_of_filters, filter_height, filter_width)
#The number of images should be 1 at all times. Depth of images will start out at 1 and then will change based on the number of filters in the previous layer, depth of filters similarly.
def do_all_layers(inputNumber, stride, padding):
    global weightMatrix
    temp = inputMatrix[inputNumber]
    #print inputMatrix.shape[0], "examples,", 
    print weightMatrix.shape[0], "layers"
    print "Input shape is", temp.shape
    for i in range(len(weightMatrix)):
        #print "Shape of weight matrix:",weightMatrix[i].shape
        #If we're not doing an FC->Conv conversion, take out this next line
        weightMatrix[i] = reshape_fc_weight_matrix(weightMatrix[i], temp.shape[1:])
        #print "Shape of weight matrix:",weightMatrix[i].shape
        #print "Number of biases (should = number of filters):",len(biasMatrix[i])
        #temp = conv_layer_forward(temp, weightMatrix[i], biasMatrix[i], stride, padding)
        temp = conv_layer_forward_ineff(temp, weightMatrix[i], biasMatrix[i], stride, padding)
        temp = relu_layer_forward(temp)
        #temp = pool_layer_forward(temp, 1)
        #temp = pool_layer_forward_ineff(temp, 1)
    #print temp
    classify(temp)

init("./example_10.txt", "./mnist_3A_layer.txt", 28, 28)
#plt.figure()
#plt.imshow(inputMatrix[0][0,0])
#plt.show()
do_all_layers(1, 1, 0)