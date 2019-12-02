#This is where we'll be doing our full implementation of a convolutional neural net with symbolic tracking.

import numpy as np;
import matplotlib.pyplot as plt
import h5py
import json
import time
import tensorflow as tf
import cPickle
import sys
import PIL.Image
from random import randint
from cStringIO import StringIO
from IPython.display import Image, display

weightMatrix = None
biasMatrix = None
inputMatrix = None
labelMatrix = None
embedInputMatrix = None
symInput = None

convWeightMatrix = None
convBiasMatrix = None
denseWeightMatrix = None
denseBiasMatrix = None
embedWeightMatrix = None

layerTypeList = []
maxPoolParams = []
activationTypeList = []
convParams = []

'''Assumed format of inputs file: list of inputs of size S+1, where the first element is discardable.
Populates inputMatrix, a matrix of N height-by-width-by-1 3D matrices, where each is an input. To make it N Sx1x1 column vecotrs, replace final nested for loops with final line. For N 1xSx1 row vectors, replace np.transpose(k) with k.'''
def read_inputs_from_file(inputFile, height, width, plusPointFive=True):
    global inputMatrix, labelMatrix
    with open(inputFile) as f:
        lines = f.readlines()
        print len(lines), "examples"
        inputMatrix = np.empty(len(lines),dtype=list)
        labelMatrix = np.zeros(len(lines),dtype=int)
        for l in range(len(lines)):
            k = [float(stringIn) for stringIn in lines[l].split(',')[1:]] #This is to remove the useless 1 at the start of each string. Not sure why that's there.
            inputMatrix[l] = np.zeros((height, width, 1),dtype=float) #we're asuming that everything is 2D for now. The 1 is just to keep numpy happy.
            labelMatrix[l] = lines[l].split(',')[0]
            count = 0
            for i in range(height):
                for j in range(width):
                    if plusPointFive:
                        inputMatrix[l][i][j] = k[count] + 0.5
                    else:
                        inputMatrix[l][i][j] = k[count]
                    count += 1
            #inputMatrix[l] = np.transpose(k) #provides Nx1 output
            
def read_cifar_inputs(inputFile):
    global inputMatrix, labelMatrix
    fo = open(inputFile, 'rb')
    pickle = cPickle.load(fo)
    data = pickle["data"]
    labels = pickle["labels"]
    inputMatrix = np.empty(data.shape[0], dtype=list)
    print data.shape[0], "examples"
    labelMatrix = labels
    for i in range(data.shape[0]):
        #inputMatrix[i] = np.zeros((32, 32, 3),dtype=uint8) #We know the dimensions for cifar
        inputMatrix[i] = data[i].astype("uint8").reshape(3,32,32).transpose([1,2,0])
        '''count = 0
        for l in range(3):
            for j in range(32):
                for k in range(32):
                    inputMatrix[i][j, k, l] = data[i, count]
                    count += 1'''
            
'''Keras models are stored in h5 files, which we can read with this function. It populates weight and bias matrices for both convolutional and fully-connected layers. It doesn't get the strides, zero-pads, architecture, or dimensions of the pooling layers. I don't know how to actively retrieve those from the .h5 file, but if we turn it into a tensorflow model.json (using tensorflowjs), they're in there; need to finish writing the function below for that. Anyway, convWeightMatrix is L entries, where L is the number of convolutional layers, each of shape n_filters x d_filter x h_filter x w_filter (unfortunately the come h x w x d x n, had to tweak that). convBiasMatrix is L entries with as many biases as the respective layer has filters, natch. denseBiasMatrix is much the same just with length M, where M is the number of FC layers, and denseWeightMatrix's M entries are input_length x output_length typical FC weight matrices, they'll need the reshape_fc_weight_matrix treatment. Alternatively, we could implement a flattening function to turn the output of the last convolutional layer into a single-dimensional output.'''
def read_weights_from_h5_file(h5File):
    global convWeightMatrix, convBiasMatrix, denseWeightMatrix, denseBiasMatrix
    global embedWeightMatrix
    f=h5py.File(h5File,'r')
    print f.keys()
    model_weights = f
    #print model_weights.keys()
    conv_layers = [p for p in model_weights.keys() if p.startswith('conv1d')]
    dense_layers = [p for p in model_weights.keys() if p.startswith('dense')]
    pool_layers = [p for p in model_weights.keys() if p.startswith('max_pool')]
    embed_layers = [p for p in model_weights.keys() if p.startswith('embedding')]
    
    convWeightMatrix  = np.empty(len(conv_layers), dtype=list)
    convBiasMatrix    = np.empty(len(conv_layers), dtype=list)
    denseWeightMatrix = np.empty(len(dense_layers),dtype=list)
    denseBiasMatrix   = np.empty(len(dense_layers),dtype=list)
    embedWeightMatrix = np.empty(len(embed_layers),dtype=list)

    layer = 0
    for k in embed_layers:
        for j in model_weights[k][k].keys():
            if j.startswith('embeddings'):
                filters = model_weights[k][k][j]
                #newShape = (filters.shape[3], filters.shape[2], filters.shape[0], filters.shape[1])
                #convWeightMatrix[layer] = np.zeros(newShape)
                #for n in range(filters.shape[3]):
                #    for h in range(filters.shape[0]):
                #        for w in range(filters.shape[1]):
                #            for d in range(filters.shape[2]):
                #                convWeightMatrix[layer][n,d,h,w] = filters[h,w,d,n]
                embedWeightMatrix[layer] = filters
                print "Embedding:{}".format(embedWeightMatrix[layer].shape)
                '''for p in range(convWeightMatrix[layer].shape[3]):
                    plt.figure()
                    plt.imshow(convWeightMatrix[layer][:,:,0,p])
                    plt.show()'''
        layer = layer+1

    layer = 0
    for k in conv_layers:
        for j in model_weights[k][k].keys():
            if j.startswith('kernel'):
                filters = model_weights[k][k][j]
                #newShape = (filters.shape[3], filters.shape[2], filters.shape[0], filters.shape[1])
                #convWeightMatrix[layer] = np.zeros(newShape)
                #for n in range(filters.shape[3]):
                #    for h in range(filters.shape[0]):
                #        for w in range(filters.shape[1]):
                #            for d in range(filters.shape[2]):
                #                convWeightMatrix[layer][n,d,h,w] = filters[h,w,d,n]
                convWeightMatrix[layer] = filters
                print "Conv Shape:{}".format(convWeightMatrix[layer].shape)
                '''for p in range(convWeightMatrix[layer].shape[3]):
                    plt.figure()
                    plt.imshow(convWeightMatrix[layer][:,:,0,p])
                    plt.show()'''
            if j.startswith('bias'):
                convBiasMatrix[layer] = np.zeros(model_weights[k][k][j].shape)
                convBiasMatrix[layer] = model_weights[k][k][j]
        layer = layer+1
    layer = 0
    for k in dense_layers:
        for j in model_weights[k][k].keys():
            if j.startswith('kernel'):
                print "Dense Shape:{}".format(model_weights[k][k][j].shape)
                denseWeightMatrix[layer] = np.zeros(model_weights[k][k][j].shape)
                denseWeightMatrix[layer] = np.array(model_weights[k][k][j], dtype=np.float)
            if j.startswith('bias'):
                denseBiasMatrix[layer] = np.zeros(model_weights[k][k][j].shape)
                denseBiasMatrix[layer] = np.array(model_weights[k][k][j], dtype=np.float)
        layer = layer+1

def parse_architecture_and_hyperparams(jsonFile):
    global layerTypeList, maxPoolParams, activationTypeList, convParams
    model = json.load(open(jsonFile))
    for layer in model['config']:
        layerTypeList.append(layer['class_name'])
        layerType = layer['class_name']
        if layerType.lower().startswith("conv"):
            #Convolutional layers should have two hyperparameters, stride and zero-padding, but I don't see zp anywhere in the model.json. We'll have to assume zero for now.
            convParams.append({'strides': layer['config']['strides']})
            activationTypeList.append(layer['config']['activation'])
        elif layerType.lower().startswith("activation"):
            activationTypeList.append(layer['config']['activation'])
        elif layerType.lower().startswith("maxpool"):
            #Max pool has two hyperparameters: stride and window size
            maxPoolParams.append({'pool_size': layer['config']['pool_size'], 'strides': layer['config']['strides']})
        elif layerType.lower().startswith("flatten"):
            pass #No hyperparameters here, and we're reshaping the dense matrices anyway.
        elif layerType.lower().startswith("dense"):
            activationTypeList.append(layer['config']['activation']) #No hyperparameters
    #print layerTypeList, maxPoolParams, activationTypeList, convParams
    
'''Annoyingly, I cant figure out how to get the structure and hyperparameters out of the graph, so we're working off of the assumption that each convolutional layer is followed by a relu and pooling layer, and that each dense layer is followed by a relu layer (except the last one). This is at least true for mnist_deep.'''
def read_weights_from_saved_tf_model(metaFile='tf_models/mnist.meta', ckpoint='./tf_models'):
    global convWeightMatrix, convBiasMatrix, denseWeightMatrix, denseBiasMatrix, layerTypeList, maxPoolParams, activationTypeList, convParams
    graph = tf.Graph()
    with tf.Session() as sess:
        imported_graph = tf.train.import_meta_graph(metaFile)
        imported_graph.restore(sess, tf.train.latest_checkpoint(ckpoint))
        graph = tf.get_default_graph()
        convLayer = 0
        denseLayer = 0
        mostRecentLayer = ""
        conv_layers = [p for p in tf.trainable_variables() if len(p.shape) == 4]
        dense_layers = [p for p in tf.trainable_variables() if len(p.shape) == 2]
        convWeightMatrix = np.empty(len(conv_layers),dtype=list)
        convBiasMatrix = np.empty(len(conv_layers),dtype=list)
        denseWeightMatrix = np.empty(len(dense_layers),dtype=list)
        denseBiasMatrix = np.empty(len(dense_layers),dtype=list)
        for v in tf.trainable_variables():
            if len(v.shape) == 4: #convolutional layer
                layerTypeList.append('conv2d')
                convWeightMatrix[convLayer] = np.zeros(v.shape)
                convWeightMatrix[convLayer] = sess.run(v)
                #print convWeightMatrix[convLayer]
                convParams.append({'strides': [1, 1]})
                mostRecentLayer = "conv"
                layerTypeList.append('activation')
                activationTypeList.append('relu')
                layerTypeList.append('maxpool')
                maxPoolParams.append({'pool_size': [2, 2], 'strides': [2, 2]})
            elif len(v.shape) == 2: #dense layer
                layerTypeList.append('dense')
                denseWeightMatrix[denseLayer] = np.zeros(v.shape)
                denseWeightMatrix[denseLayer] = sess.run(v)
                mostRecentLayer = "dense"
                layerTypeList.append('activation')
                activationTypeList.append('relu')
            elif len(v.shape) == 1: #bias
                if(mostRecentLayer == "conv"):
                    convBiasMatrix[convLayer] = np.zeros(v.shape)
                    convBiasMatrix[convLayer] = sess.run(v)
                    convLayer = convLayer + 1
                elif(mostRecentLayer == "dense"):
                    denseBiasMatrix[denseLayer] = np.zeros(v.shape)
                    denseBiasMatrix[denseLayer] = sess.run(v)
                    denseLayer = denseLayer + 1
                
            print v.shape
        if(layerTypeList[-1] == 'activation'):
            layerTypeList[-1] = "" #Removes last activation layer.
        sess.close()
    
'''Assumed format of weights file (see mnist_3A_layer.txt for example): [N=numberOfLayers\n\n(X=heightOfFilter,Y=widthOfFilter\n(Z=csvOfXTimesYWeights)\n(B=csvOfYBiases)\n\n)*N]
Populates weightMatrix, a 3D matrix of N 2D matrices with X rows and Y columns, and biasMatrix, a 2D matrix of N rows where each row is a list of Y biases for the respective filters '''
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
    #N, C, H, W = input_shape
    #print "N, C, H, W: "+N+" "+C+" "+H+" "+W
    C, H, W = input_shape
    print "C, H, W: ",C,H,W
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1
    
    print out_height, out_width
    
    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    print i0
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    
    return (k, i, j)
    
'''We're assuming the image is 2D for now.'''
def im2col_indices(input, section_height, section_width, padding=1, stride=1):
    input_padded = np.pad(input, ((padding, padding),(padding, padding)), mode='constant')
    k, i, j = get_im2col_indices((1,)+input.shape, section_height, section_width, padding, stride)
    print k, i, j
    cols = input_padded[:, k, i, j]
    print cols
    C = input.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(section_height * section_width * C, -1)
    return cols

def im2col_sliding_strided(A, block_shape, stepsize=1, padding=0):
    m,n,d = A.shape
    A_padded = np.pad(A, ((padding, padding),(padding, padding),(0,0)), mode='constant')
    #print "Im2col input shape:",m, n
    s0, s1, s2 = A_padded.strides   
    #print "Im2col input strides",s0, s1 
    nrows = m-block_shape[0]+1
    ncols = n-block_shape[1]+1
    #print "Im2col output # of rows, cols:",nrows, ncols
    shp = block_shape[0],block_shape[1],nrows,ncols
    #print shp
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(A_padded, shape=shp, strides=strd)
    return out_view.reshape(block_shape[0]*block_shape[1],-1)[:,::stepsize]

def conv_layer_forward_ineff(input, filters, biases, stride=1, padding=1, keras=False):
    print "Beginning conv layer"
    h_x, w_x, d_x = input.shape
    if(keras):
        print filters.shape
        h_filter, w_filter, n_filters = filters.shape #d_x should equal d_filter
        d_filter = 1
    else:
        n_filters, d_filter, h_filter, w_filter = filters.shape
    if(padding == -1):
        padding = (h_filter-1)/2
    input_padded = np.pad(input,((padding, padding),(padding, padding),(0,0)),mode='constant')
    print input.shape, filters.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1
    print h_out, "x", w_out, "out"
    out = np.zeros((h_out, w_out, n_filters))
    #print "Biases:",biases
    #print input_padded[4,0:w_filter,0]
    #print filters[0,0,0:h_filter,0:w_filter]
    for i in range(n_filters):
        #print "Applying filter", i
        #print "Bias:", biases[i]
        for j in range(h_out):
            for k in range(w_out):
                rowIndex = j*stride
                colIndex = k*stride
                for l in range(d_x):
                    #print filters[i,l].shape
                    #print input_padded[j:j+h_filter,k:k+w_filter,l].shape
                    if(keras):
                        '''if(i == 29):
                            print filters[:,:,l,i]
                            print biases[i]
                            print input_padded[j:j+h_filter,k:k+w_filter,l]
                            plt.figure()
                            plt.imshow(input_padded[rowIndex:rowIndex+h_filter, colIndex:colIndex+w_filter,l])
                            plt.show()'''
                        out[j,k,i] = out[j,k,i] + np.sum(np.multiply(filters[:,:,i], input_padded[rowIndex:rowIndex+h_filter,colIndex:colIndex+w_filter,l]))
                    else:
                        out[j,k,i] = out[j,k,i] + np.sum(np.multiply(filters[i,l], input_padded[rowIndex:rowIndex+h_filter,colIndex:colIndex+w_filter,l]))
                out[j,k,i] = out[j,k,i] + biases[i]
                '''if out[j,k,i] > 0.12:
                    if i == 29'''
                #print out[j,k,i]
        #print out[:,:,i]
    print "output shape",out.shape,"\n"
    out = out.reshape(h_out, n_filters, w_out)
    return out

def relu_layer_forward(x):
    print "Beginning relu layer"
    relu = lambda x: x * (x > 0).astype(float) #PC = PC ^ x > 0
    return relu(x);
    print ""

'''This is nonfunctional for the same reasons as conv_layer_forward above.
We could make this multi-channel with an "n" parameter, changes needed noted in comments, assuming we also changed the way we were shaping things to d x w x h'''
def pool_layer_forward(X, size, stride = 1):
    print "Beginning pool layer"
    #print "X shape:",X.shape
    h, w, d = X.shape
    h_out = (h-size)/stride + 1
    w_out = (w-size)/stride + 1
    X_reshaped = X.reshape(h, w, d) #(n*d, 1, h, w)
    X_col = im2col_sliding_strided(X_reshaped, (size, size), stepsize=stride) #im2col_indices(X_reshaped, size, size, padding=0, stride=stride)
    #print "X_col shape:",X_col.shape
    max_idx = np.argmax(X_col, axis=0)
    out = X_col[max_idx, range(max_idx.size)]
    out = out.reshape(h_out, w_out, d) #(h_out, w_out, n, d)
    #out = out.transpose(2, 3, 0, 1)
    print ""
    return out;
    
def pool_layer_forward_ineff(X, size, stride = 1):
    print "Beginning pool layer"
    #print "X shape:",X.shape
    h, w, d = X.shape
    h_out = (h-size)/stride + 1
    w_out = (w-size)/stride + 1
    #X_reshaped = X.reshape(h, w, d)
    out = np.zeros((h_out, w_out, d))
    for i in range(h_out):
        for j in range(w_out):
            for k in range(d):
                rowIndex = i*stride
                colIndex = j*stride
                out[i,j,k] = X[rowIndex:rowIndex+size,colIndex:colIndex+size,k].max()
    print ""
    return out
    
def concolic_pool_layer_forward(X, size, stride = 1):
    global symInput
    print "Beginning concolic pool layer"
    h, w, d = X.shape
    h_out = (h-size)/stride + 1
    w_out = (w-size)/stride + 1
    out = np.zeros((h_out, w_out, d))
    symOut = np.zeros((h_out, w_out, d, symInput.shape[3], symInput.shape[4]))
    for i in range(h_out):
        for j in range(w_out):
            for k in range(d):
                rowIndex = i*stride
                colIndex = j*stride
                #print X[rowIndex:rowIndex+size,colIndex:colIndex+size,k], X[owIndex:rowIndex+size,j:j+size,k].max()
                max_idx = np.argmax(X[rowIndex:rowIndex+size,colIndex:colIndex+size,k])
                max_row = max_idx/size
                max_col = max_idx % size
                #print symInput[i:i+size,j:j+size,k][max_row, max_col].shape
                symOut[i,j,k] = symInput[rowIndex:rowIndex+size,colIndex:colIndex+size,k][max_row, max_col]
                out[i,j,k] = X[rowIndex:rowIndex+size,colIndex:colIndex+size,k].max()
    symInput = symOut
    print ""
    return out
    
def concolic_pool_layer_forward_3d(X, size, stride = 1):
    global symInput
    print "Beginning 3D concolic pool layer"
    h, w, d = X.shape
    h_out = (h-size)/stride + 1
    w_out = (w-size)/stride + 1
    out = np.zeros((h_out, w_out, d))
    symOut = np.zeros((h_out, w_out, d, symInput.shape[3], symInput.shape[4], 3))
    for i in range(h_out):
        for j in range(w_out):
            for k in range(d):
                rowIndex = i*stride
                colIndex = j*stride
                #print X[rowIndex:rowIndex+size,colIndex:colIndex+size,k], X[owIndex:rowIndex+size,j:j+size,k].max()
                max_idx = np.argmax(X[rowIndex:rowIndex+size,colIndex:colIndex+size,k])
                max_row = max_idx/size
                max_col = max_idx % size
                #print symInput[rowIndex:rowIndex+size,colIndex:colIndex+size,k][max_row, max_col].shape
                symOut[i,j,k] = symInput[rowIndex:rowIndex+size,colIndex:colIndex+size,k][max_row, max_col]
                out[i,j,k] = X[rowIndex:rowIndex+size,colIndex:colIndex+size,k].max()
    symInput = symOut
    print ""
    return out

'''Each node in the layer must have an array with a list input.size of coefficients. 
If our input is X x Y, every point on a given internal/output layer needs an X x Y map of coefficients.
The simple way of doing it: we start off with a ([n x] h x w x d x h x w) input with 1's in all the different h x w locations. That turns into an ([n x] h_out x w_out x n_filters x h x w) map with each passing layer. Each location on that map is the sum of h_filter x w_filter maps from a particular stride, each multiplied by the appropriate filter value. '''
def sym_conv_layer_forward(input, filters, b, stride=1, padding=1, keras=False):
    print "Beginning sym conv layer"
    h_prev, w_prev, d_prev, h_x, w_x = input.shape
    if(keras):
        h_filter, w_filter, n_filters = filters.shape #d_x should equal d_filter
        d_filter = 1
    else:
        n_filters, d_filter, h_filter, w_filter = filters.shape
    if padding == -1:
        padding = (h_filter - 1)/2
    h_out = (h_prev - h_filter + 2 * padding) / stride + 1
    w_out = (w_prev - w_filter + 2 * padding) / stride + 1
    input_padded = np.pad(input,((padding, padding),(padding, padding),(0,0),(0,0),(0,0)),mode='constant')
    #print "Padded input shape:", input_padded.shape, "filters shape:", filters.shape
    out = np.zeros((h_out, w_out, n_filters, h_x, w_x))
    for i in range(n_filters):
        #print "Applying sym filter", i
        for j in range(h_out):
            for k in range(w_out):
                rowIndex = j*stride
                colIndex = k*stride
                temp = np.zeros((h_x, w_x))
                for l in range(d_prev):
                    #print filters[i,l].shape
                    #print input_padded[j:j+h_filter,k:k+w_filter,l].shape
                    #temp = np.zeros((h_x, w_x))
                    for m in range(h_filter):
                        for n in range(w_filter):
                            if(keras):
                                scaledMatrix = np.multiply(filters[m,n,i], input_padded[rowIndex+m,colIndex+n,l])
                            else:
                                scaledMatrix = np.multiply(filters[i,l,m,n], input_padded[rowIndex+m,colIndex+n,l])
                            temp = np.add(temp, scaledMatrix)
                out[j,k,i] = temp
                #out[j,k,i] = np.add(out[j,k,i], b[i])
    
    print "Output shape:", out.shape
    print ""
    out = out.reshape(h_out, n_filters, w_out, h_x, w_x)
    return out
    
def sym_conv_layer_forward_3d(input, filters, b, stride=1, padding=1, keras=False):
    print "Beginning sym conv layer"
    h_prev, w_prev, d_prev, h_x, w_x, d_x = input.shape
    if(keras):
        h_filter, w_filter, d_filter, n_filters = filters.shape #d_x should equal d_filter
    else:
        n_filters, d_filter, h_filter, w_filter = filters.shape
    if padding == -1:
        padding = (h_filter - 1)/2
    h_out = (h_prev - h_filter + 2 * padding) / stride + 1
    w_out = (w_prev - w_filter + 2 * padding) / stride + 1
    input_padded = np.pad(input,((padding, padding),(padding, padding),(0,0),(0,0),(0,0),(0,0)),mode='constant')
    #print "Padded input shape:", input_padded.shape, "filters shape:", filters.shape
    out = np.zeros((h_out, w_out, n_filters, h_x, w_x, d_x))
    for i in range(n_filters):
        #print "Applying sym filter", i
        for j in range(h_out):
            for k in range(w_out):
                rowIndex = j*stride
                colIndex = k*stride
                temp = np.zeros((h_x, w_x, d_x))
                for l in range(d_prev):
                    #print filters[i,l].shape
                    #print input_padded[j:j+h_filter,k:k+w_filter,l].shape
                    #temp = np.zeros((h_x, w_x))
                    for m in range(h_filter):
                        for n in range(w_filter):
                            if(keras):
                                scaledMatrix = np.multiply(filters[m,n,l,i], input_padded[rowIndex+m,colIndex+n,l])
                            else:
                                scaledMatrix = np.multiply(filters[i,l,m,n], input_padded[rowIndex+m,colIndex+n,l])
                            temp = np.add(temp, scaledMatrix)
                out[j,k,i] = temp
                #out[j,k,i] = np.add(out[j,k,i], b[i])
        #plt.figure()
        #plt.imshow(np.sum(np.sum(np.sum(out[:,:,i], axis=0), axis=0),axis=2))
        #plt.show()
    
    print "Output shape:", out.shape
    print ""
    return out
    
'''This was my first attempt at making a symbolic convolutional forward pass, based on the original implementation of conv_forward. Keeping it around for now just in case.'''
def unused_sym_conv(input, filter, b, stride=1, padding=1):
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
    
def init_symInput(inputHeight, inputWidth):
    global symInput
    symInput = np.zeros((inputHeight, inputWidth, 1, inputHeight, inputWidth))
    for i in range(inputHeight):
        for j in range(inputWidth):
            symInput[i,j,0,i,j] = 1
            
def init_3d_symInput(inputHeight, inputWidth):
    global symInput
    symInput = np.zeros((inputHeight, inputWidth, 3, inputHeight, inputWidth, 3))
    for i in range(inputHeight):
        for j in range(inputWidth):
            for k in range(3):
                symInput[i,j,k,i,j,k] = 1

def init(inputFile, weightFile, inputHeight, inputWidth, plusPointFive=True):
    global symInput
    read_inputs_from_file(inputFile, inputHeight, inputWidth, plusPointFive)
    read_weights_from_file(weightFile)
    init_symInput(inputHeight, inputWidth)
    
'''This assumes that output is of form d x h x w. We don't do that, so this is just in case we bother to get conv_layer_forward working.'''
def classify(processedArray):
    maxValue = 0
    maxIndex = -1
    for i in range(len(processedArray)):
        print "Class",i,"confidence",processedArray[i,0,0]
        if(processedArray[i,0,0] > maxValue):
            maxValue = processedArray[i,0,0]
            maxIndex = i
    print "MaxIndex:",maxIndex
    
def classify_ineff(processedArray):
    maxValue = -sys.maxint -1
    maxIndex = -1
    for i in range(processedArray.shape[2]):
        print "Class",i,"confidence",processedArray[0,0,i]
        if(processedArray[0,0,i] > maxValue):
            maxValue = processedArray[0,0,i]
            maxIndex = i
    print "MaxIndex:",maxIndex
    #print symInput[0,0,maxIndex]
    return maxIndex
  
'''Each column of an FC weight matrix is a filter that will be placed over the entire input. We want to turn each of them into a proper_height x proper_width matrix, and end up with something of shape (proper_height, proper_width, n_filters). ''' 
def reshape_fc_weight_matrix_keras(fcWeights, proper_shape):
    total_height, n_filters = fcWeights.shape
    proper_height, proper_width, proper_depth = proper_shape
    '''temp = np.zeros((proper_height, proper_width, proper_depth, n_filters))
    for i in range(n_filters):
        for j in range(proper_depth):
            for k in range(proper_height):
                for l in range(proper_width):
                    index = k*proper_width + l + j
                    temp[k,l,j,i] = fcWeights[index,i]'''
    return fcWeights.reshape((proper_height, proper_width, n_filters))

'''Each column of an FC weight matrix is a filter that will be placed over the entire input. We want to turn each of them into a proper_height x proper_width matrix, and end up with something of shape (n_filters, proper_height, proper_width) (slightly different from the keras version above).'''   
def reshape_fc_weight_matrix(fcWeights, proper_shape):
    total_height, n_filters = fcWeights.shape
    proper_height, proper_width, proper_depth = proper_shape
    temp = np.zeros((n_filters, proper_depth, proper_height, proper_width))
    
    for i in range(n_filters):
        for j in range(proper_depth):
            for k in range(proper_height):
                for l in range(proper_width):
                    index = k*proper_width + l + j
                    temp[i,j,k,l] = fcWeights[index,i]
    return temp

def inspect_sym_input(inputImage):
    for i in range(symInput.shape[2]):
        thing = np.zeros((symInput.shape[3],symInput.shape[4]))
        for j in range(symInput.shape[0]):
            for k in range(symInput.shape[1]):
                thing = np.add(thing, symInput[j,k,i])
        plt.figure()
        plt.imshow(normalize_to_255(np.multiply(thing, inputImage[:,:,0])))
        plt.title("Sym input at node %d"%i)
        plt.show()
        
def inspect_3d_sym_input():
    for i in range(symInput.shape[2]):
        thing = np.zeros((symInput.shape[3],symInput.shape[4],symInput.shape[5]))
        for j in range(symInput.shape[0]):
            for k in range(symInput.shape[1]):
                thing = np.add(thing, symInput[j,k,i])
        for l in range(symInput.shape[5]):
            plt.figure()
            plt.imshow(thing[:,:,l])
            plt.title("Sym input at node %d, color %d" % (i, l))
            plt.show()
        
def inspect_intermediate_output(temp):
    for i in range (temp.shape[2]):
        thing = np.zeros((temp.shape[0],temp.shape[1]))
        for j in range(temp.shape[0]):
            for k in range(temp.shape[1]):
                thing = np.add(thing, temp[:,:,i])
        plt.figure()
        plt.imshow(thing)
        plt.show()
        
def get_top_pixels(x, percent):
    temp = x.flatten()
    top_values = np.unique(temp)[-int(len(np.unique(temp)) * percent):]
    print "Returning", int(len(np.unique(temp)) * percent), "pixels"
    for i in range(len(temp)):
        if temp[i] not in top_values:
            temp[i] = 0
    return temp.reshape(x.shape)
    
def get_above_average_pixels(x):
    temp = x.flatten()
    average = np.average(np.unique(x.flatten()))
    for i in range(len(temp)):
        if not (temp[i] >= average):
            temp[i] = 0
    return temp.reshape(x.shape)
    
def get_most_different_pixels(x, y):
    xTemp = x.flatten()
    yTemp = y.flatten()
    temp = np.zeros(xTemp.shape)
    for i in range(len(xTemp)):
        temp[i] = abs(xTemp[i] - yTemp[i])
    temp = temp.reshape(x.shape)
    return temp
    
def compare_pixel_ranks(x, y, tolerance=0):
    temp1 = x.flatten()
    temp2 = y.flatten()
    top_indices_1 = np.argsort(temp1)
    top_indices_2 = np.argsort(temp2)
    equal_locations = 0
    for i in range(len(top_indices_1)):
        #if top_indices_1[i] <= top_indices_2[i]+tolerance and top_indices_1[i] >= top_indices_2[i]-tolerance:
        if abs(top_indices_1[i] - top_indices_2[i]) <= tolerance:
            equal_locations += 1
    print "Ranks are equal at", equal_locations, "spots"
    return equal_locations
    
def image_based_on_pixel_ranks(x):
    temp = x.flatten()
    sortIndices = temp.argsort()
    ranks = np.empty_like(sortIndices)
    ranks[sortIndices] = np.arange(len(temp))
    return ranks.reshape(x.shape)
    
def write_pixel_ranks_to_file(x, filename):
    temp = x.flatten()
    sortIndices = temp.argsort()
    ranks = np.empty_like(sortIndices)
    ranks[sortIndices] = np.arange(len(temp))
    ranks = ranks.reshape(x.shape)
    write_image_to_file(ranks, filename)
    '''with open(filename, "w") as f:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                f.write("%d\t" % ranks[i,j])
            f.write("\n")'''
            
def write_image_to_file(x, filename):
    with open(filename, "w") as f:
        if len(x.shape) == 2:
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    f.write("%f\t" % x[i,j])
                f.write("\n")
        else:
            for k in range(x.shape[2]):
                for i in range(x.shape[0]):
                    for j in range(x.shape[1]):
                        f.write("%f\t" % x[i,j,k])
                    f.write("\n")
                f.write("\n")
            
def write_image_to_file_scientific(x, filename):
    with open(filename, "w") as f:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                f.write("%E\t" % x[i,j])
            f.write("\n")
    
def normalize_to_255(x):
    temp = x.flatten()
    maximum = np.amax(temp)
    minimum = np.amin(temp)
    norm = np.multiply(np.array([(i - minimum) / (maximum - minimum) for i in temp]), 255)
    '''for i in range (len(temp)):
        temp[i] = temp[i]/maximum * 255'''
    return norm.reshape(x.shape)
    
def normalize_to_1(x):
    temp = x.flatten()
    maximum = np.amax(temp)
    minimum = np.amin(temp)
    norm = np.multiply(np.array([(i - minimum) / (maximum - minimum) for i in temp]), 1)
    return norm.reshape(x.shape)
    
def gray_scale(img):
    '''Converts the provided RGB image to gray scale.
    '''
    img = np.average(img, axis=2)
    return np.transpose([img, img, img], axes=[1,2,0])
  
def pil_img(a):
    '''Returns a PIL image created from the provided RGB array.
    '''
    a = np.uint8(a)
    return PIL.Image.fromarray(a)
  
def show_img(img, fmt='jpeg'):
    img.show()
    '''Displays the provided PIL image
    '''
    '''f = StringIO()
    img.save(f, fmt)
    display(Image(data=f.getvalue()))'''
  
def visualize_attrs_windowing(img, attrs, ptile=99):
    '''Visaualizes the provided attributions by first aggregating them
    along the color channel to obtain per-pixel attributions and then
    scaling the intensities of the pixels in the original image in
    proportion to absolute value of these attributions.

    The provided image and attributions must of shape (224, 224, 3).
    '''
    attrs = gray_scale(attrs)
    attrs = abs(attrs)
    attrs = np.clip(attrs/np.percentile(attrs, ptile), 0,1)
    vis = img*attrs
    show_img(pil_img(vis))
    return pil_img(vis)
    
def do_all_layers_for_image(squareImage, padding=0, stride=1):
    global weightMatrix, symInput
    temp = squareImage
    print inputMatrix.shape, weightMatrix.shape
    print "Input shape is", temp.shape
    for i in range(len(weightMatrix)):
        #print "Shape of weight matrix:",weightMatrix[i].shape
        #If we're not doing an FC->Conv conversion, take out this next line
        weightMatrix[i] = reshape_fc_weight_matrix(weightMatrix[i], temp.shape)
        #weightMatrix[i] = reshape_fc_weight_matrix_keras(weightMatrix[i], temp.shape)
        #print "Shape of weight matrix:",weightMatrix[i].shape
        temp = conv_layer_forward_ineff(temp, weightMatrix[i], biasMatrix[i], stride, padding)
        if(i != len(weightMatrix)-1):
            temp = relu_layer_forward(temp)
        symInput = sym_conv_layer_forward(symInput, weightMatrix[i], biasMatrix[i], stride, padding)
        #symInput = relu_layer_forward(symInput)
        #temp = pool_layer_forward_ineff(temp, 1)
        temp = concolic_pool_layer_forward(temp, 1)
        #print temp
    print symInput.shape
    #classify(temp)
    #plt.imshow(inputMatrix[inputNumber][:,:,0])
    maxIndex = classify_ineff(temp)
    return maxIndex
    
def do_all_layers_keras(inputNumber, outDir):
    global symInput, convWeightMatrix, denseWeightMatrix, embedInputMatrix
    temp = inputMatrix[inputNumber]
    convIndex = 0
    denseIndex = 0
    poolIndex = 0
    activationIndex = 0
    print(layerTypeList)
    print(temp.shape)
    for layerType in layerTypeList:
        if layerType.lower().startswith("embedding"):
            temp2 = []
            for x in temp:
                x = int(x[0][0])
                embed = embedWeightMatrix[0][x]
		temp2.append(embed)
            temp = np.array(temp2)
            print temp.shape
            h,w = temp.shape
            temp = temp.reshape((h,w,1))
            embedInputMatrix = np.copy(temp)
            print temp.shape
        if layerType.lower().startswith("conv"):
            '''if convIndex == 1:
                continue'''
            #print convWeightMatrix[convIndex], convBiasMatrix[convIndex], convParams[convIndex]['strides'][0]
            # When using the keras file: padding of zero. When using mnist_deep: -1.
            temp = conv_layer_forward_ineff(temp, convWeightMatrix[convIndex], convBiasMatrix[convIndex], convParams[convIndex]['strides'][0], 0, keras=True)
            symInput = sym_conv_layer_forward(symInput, convWeightMatrix[convIndex], convBiasMatrix[convIndex], convParams[convIndex]['strides'][0], 0, keras=True)
            convIndex = convIndex + 1
            #inspect_intermediate_output(temp)
            #inspect_sym_input()
        elif layerType.lower().startswith("activation"):
            activationType = activationTypeList[activationIndex].lower()
            if activationType == 'relu':
                np.set_printoptions(threshold=np.nan)
                temp = relu_layer_forward(temp)
                '''for i in range(temp.shape[0]):
                    for j in range(temp.shape[1]):
                        print temp[i, j]'''
                '''for i in range(temp.shape[2]):
                    print temp[:, :, i]'''
            if activationType == 'linear':
                continue
                #symInput = relu_layer_forward(symInput)
            activationIndex = activationIndex + 1
        elif layerType.lower().startswith("maxpool"):
            #inspect_intermediate_output(temp)
            #inspect_sym_input()
            #temp = pool_layer_forward_ineff(temp, maxPoolParams[poolIndex]['pool_size'][0], maxPoolParams[poolIndex]['strides'][0])
            temp = concolic_pool_layer_forward(temp, maxPoolParams[poolIndex]['pool_size'][0], maxPoolParams[poolIndex]['strides'][0])
            #inspect_intermediate_output(temp)
            #inspect_sym_input()
            poolIndex = poolIndex + 1 
        elif layerType.lower().startswith("flatten"):
            pass
        elif layerType.lower().startswith("dense"):
            tempWeightMatrix = reshape_fc_weight_matrix_keras(denseWeightMatrix[denseIndex], temp.shape)
            temp = conv_layer_forward_ineff(temp, tempWeightMatrix, denseBiasMatrix[denseIndex], 1, 0, keras=True)
            symInput = sym_conv_layer_forward(symInput, tempWeightMatrix, denseBiasMatrix[denseIndex], 1, 0, keras=True)
            #inspect_sym_input(inputMatrix[inputNumber])
            denseIndex = denseIndex + 1
    maxIndex = classify_ineff(temp);
    print('Concolic label predicted:{}'.format(maxIndex))


    #Coeffs, coeffs*input
    '''plt.figure()
    plt.imshow(inputMatrix[inputNumber][:,:,0])'''
    #plt.figure()
    #plt.imshow(normalize_to_255(symInput[0,0,maxIndex]))
    #plt.savefig('./result_images/coefficient_attributions/%s/coefficients/%s_sym_coeffs_%d' % (outDir, outDir, inputNumber))
    #write_image_to_file(symInput[0,0,maxIndex], './result_images/coefficient_attributions/%s/coefficients/%s_sym_coeffs_%d.txt' % (outDir, outDir, inputNumber))
    #plt.figure()
    #plt.imshow(normalize_to_255(np.multiply(symInput[0,0,inputNumber], inputMatrix[inputNumber][:,:,0])))
    #plt.savefig('./result_images/coefficient_attributions/%s/coefficients_times_input/%s_sym_coeffs_%d_mult_input'% (outDir, outDir, inputNumber))
    #write_image_to_file(np.multiply(symInput[0,0,inputNumber], inputMatrix[inputNumber][:,:,0]), './result_images/coefficient_attributions/%s/coefficients_times_input/%s_sym_coeffs_%d_mult_input.txt'% (outDir, outDir, inputNumber))
    #plt.close()
    #plt.show()
    
    #Pixel ranks of the above
    #plt.figure()
    #plt.imshow(image_based_on_pixel_ranks(symInput[0,0,maxIndex]))
    #plt.savefig('./result_images/coefficient_attributions/%s/coefficients/Pixel_ranks/%s_ranked_sym_coeffs_%d' % (outDir, outDir, inputNumber))
    #write_pixel_ranks_to_file(symInput[0,0,maxIndex], './result_images/coefficient_attributions/%s/coefficients/Pixel_ranks/%s_sym_coeffs_ranks_%d.txt' % (outDir, outDir, inputNumber))
    #plt.figure()
    #plt.imshow(image_based_on_pixel_ranks(np.multiply(symInput[0,0,maxIndex], inputMatrix[inputNumber][:,:,0])))
    #plt.savefig('./result_images/coefficient_attributions/%s/coefficients_times_input/Pixel_ranks/%s_ranked_sym_coeffs_times_in_%d' % (outDir, outDir, inputNumber))
    #write_pixel_ranks_to_file(np.multiply(symInput[0,0,maxIndex], inputMatrix[inputNumber][:,:,0]), './result_images/coefficient_attributions/%s/coefficients_times_input/Pixel_ranks/%s_sym_coeffs_times_in_ranks_%d.txt' % (outDir, outDir, inputNumber))
    #plt.show()
    #inspect_sym_input()
    #if maxIndex != labelMatrix[inputNumber]:
    #    print "Error, correct label is", labelMatrix[inputNumber]
    return maxIndex
    
def do_all_layers_keras_for_image(squareImage):
    global symInput, convWeightMatrix, denseWeightMatrix
    temp = squareImage
    convIndex = 0
    denseIndex = 0
    poolIndex = 0
    activationIndex = 0
    for layerType in layerTypeList:
        if layerType.lower().startswith("conv"):
            '''if convIndex == 1:
                continue'''
            #print convWeightMatrix[convIndex], convBiasMatrix[convIndex], convParams[convIndex]['strides'][0]
            temp = conv_layer_forward_ineff(temp, convWeightMatrix[convIndex], convBiasMatrix[convIndex], convParams[convIndex]['strides'][0], 0, keras=True)
            symInput = sym_conv_layer_forward(symInput, convWeightMatrix[convIndex], convBiasMatrix[convIndex], convParams[convIndex]['strides'][0], 0, keras=True)
            convIndex = convIndex + 1
            #inspect_intermediate_output(temp)
            #inspect_sym_input()
        elif layerType.lower().startswith("activation"):
            activationType = activationTypeList[activationIndex].lower()
            if activationType == 'relu':
                np.set_printoptions(threshold=np.nan)
                temp = relu_layer_forward(temp)
                '''for i in range(temp.shape[0]):
                    for j in range(temp.shape[1]):
                        print temp[i, j]'''
                '''for i in range(temp.shape[2]):
                    print temp[:, :, i]'''
                #symInput = relu_layer_forward(symInput)
            activationIndex = activationIndex + 1
        elif layerType.lower().startswith("maxpool"):
            #inspect_intermediate_output(temp)
            #inspect_sym_input()
            #temp = pool_layer_forward_ineff(temp, maxPoolParams[poolIndex]['pool_size'][0], maxPoolParams[poolIndex]['strides'][0])
            temp = concolic_pool_layer_forward(temp, maxPoolParams[poolIndex]['pool_size'][0], maxPoolParams[poolIndex]['strides'][0])
            #inspect_intermediate_output(temp)
            #inspect_sym_input()
            poolIndex = poolIndex + 1
        elif layerType.lower().startswith("flatten"):
            pass
        elif layerType.lower().startswith("dense"):
            tempWeightMatrix = reshape_fc_weight_matrix_keras(denseWeightMatrix[denseIndex], temp.shape)
            temp = conv_layer_forward_ineff(temp, tempWeightMatrix, denseBiasMatrix[denseIndex], 1, 0, keras=True)
            symInput = sym_conv_layer_forward(symInput, tempWeightMatrix, denseBiasMatrix[denseIndex], 1, 0, keras=True)
            denseIndex = denseIndex + 1
    maxIndex = classify_ineff(temp)
    return maxIndex

def do_all_layers_keras_3d(inputNumber, outDir):
    global symInput, convWeightMatrix, denseWeightMatrix
    temp = inputMatrix[inputNumber]
    convIndex = 0
    denseIndex = 0
    poolIndex = 0
    activationIndex = 0
    for layerType in layerTypeList:
        if layerType.lower().startswith("conv"):
            '''if convIndex == 1:
                continue'''
            #print convWeightMatrix[convIndex], convBiasMatrix[convIndex], convParams[convIndex]['strides'][0]
            # When using the keras file: padding of zero. When using mnist_deep: -1.
            temp = conv_layer_forward_ineff(temp, convWeightMatrix[convIndex], convBiasMatrix[convIndex], convParams[convIndex]['strides'][0], 0, keras=True)
            symInput = sym_conv_layer_forward_3d(symInput, convWeightMatrix[convIndex], convBiasMatrix[convIndex], convParams[convIndex]['strides'][0], 0, keras=True)
            convIndex = convIndex + 1
            #inspect_intermediate_output(temp)
            #inspect_3d_sym_input()
        elif layerType.lower().startswith("activation"):
            activationType = activationTypeList[activationIndex].lower()
            if activationType == 'relu':
                np.set_printoptions(threshold=np.nan)
                temp = relu_layer_forward(temp)
                '''for i in range(temp.shape[0]):
                    for j in range(temp.shape[1]):
                        print temp[i, j]'''
                '''for i in range(temp.shape[2]):
                    print temp[:, :, i]'''
                #symInput = relu_layer_forward(symInput)
            activationIndex = activationIndex + 1
        elif layerType.lower().startswith("maxpool"):
            #inspect_intermediate_output(temp)
            #inspect_3d_sym_input()
            #temp = pool_layer_forward_ineff(temp, maxPoolParams[poolIndex]['pool_size'][0], maxPoolParams[poolIndex]['strides'][0])
            temp = concolic_pool_layer_forward_3d(temp, maxPoolParams[poolIndex]['pool_size'][0], maxPoolParams[poolIndex]['strides'][0])
            #inspect_intermediate_output(temp)
            #inspect_3d_sym_input()
            poolIndex = poolIndex + 1 
        elif layerType.lower().startswith("flatten"):
            pass
        elif layerType.lower().startswith("dense"):
            tempWeightMatrix = reshape_fc_weight_matrix_keras(denseWeightMatrix[denseIndex], temp.shape)
            temp = conv_layer_forward_ineff(temp, tempWeightMatrix, denseBiasMatrix[denseIndex], 1, 0, keras=True)
            symInput = sym_conv_layer_forward_3d(symInput, tempWeightMatrix, denseBiasMatrix[denseIndex], 1, 0, keras=True)
            #inspect_3d_sym_input()
            denseIndex = denseIndex + 1
    maxIndex = classify_ineff(temp);
    #symResultImage = np.sum(symInput[0,0,maxIndex], axis=2)
    symResultImage = symInput[0,0,maxIndex]
    symTimesIn = np.zeros((32,32,3))
    for i in range(3):
        symTimesIn[:,:,i] = np.multiply(symInput[0,0,maxIndex,:,:,i], inputMatrix[inputNumber][:,:,i])
        '''plt.figure()
        plt.imshow(symInput[0,0,maxIndex,:,:,i])
        plt.title("Coeffs, color number %d"%i)
        plt.show()
        plt.imshow(symTimesIn[:,:,i])
        plt.title("coeffs*in, color number %d"%i)
        plt.show()'''
    #symTimesInImage = np.sum(symTimesIn, axis=2)
    symTimesInImage = visualize_attrs_windowing(inputMatrix[inputNumber], symInput[0,0,maxIndex])
    
    #Coeffs, coeffs*input
    '''plt.figure()
    plt.imshow(inputMatrix[inputNumber][:,:,0])'''
    plt.figure()
    plt.imshow(np.sum(symResultImage, axis=2))
    plt.savefig('./result_images/coefficient_attributions/%s/coefficients/%s_sym_coeffs_%d' % (outDir, outDir, inputNumber))
    write_image_to_file(symInput[0,0,maxIndex], './result_images/coefficient_attributions/%s/coefficients/%s_sym_coeffs_%d.txt' % (outDir, outDir, inputNumber))
    '''plt.figure()
    plt.imshow(normalize_to_255(symTimesInImage))
    plt.savefig('./result_images/coefficient_attributions/%s/coefficients_times_input/%s_sym_coeffs_%d_mult_input'% (outDir, outDir, inputNumber))
    write_image_to_file(symTimesIn, './result_images/coefficient_attributions/%s/coefficients_times_input/%s_sym_coeffs_%d_mult_input.txt'% (outDir, outDir, inputNumber))'''
    symTimesInImage.save('./result_images/coefficient_attributions/%s/coefficients_times_input/%s_sym_coeffs_%d_mult_input.png'% (outDir, outDir, inputNumber))
    #symTimesInImage.show()
    
    #Pixel ranks of the above
    plt.figure()
    plt.imshow(image_based_on_pixel_ranks(symResultImage))
    plt.savefig('./result_images/coefficient_attributions/%s/coefficients/Pixel_ranks/%s_ranked_sym_coeffs_%d' % (outDir, outDir, inputNumber))
    write_pixel_ranks_to_file(symInput[0,0,maxIndex], './result_images/coefficient_attributions/%s/coefficients/Pixel_ranks/%s_sym_coeffs_ranks_%d.txt' % (outDir, outDir, inputNumber))
    '''plt.figure()
    plt.imshow(image_based_on_pixel_ranks(symTimesInImage))
    plt.savefig('./result_images/coefficient_attributions/%s/coefficients_times_input/Pixel_ranks/%s_ranked_sym_coeffs_times_in_%d' % (outDir, outDir, inputNumber))'''
    write_pixel_ranks_to_file(symTimesIn, './result_images/coefficient_attributions/%s/coefficients_times_input/Pixel_ranks/%s_sym_coeffs_times_in_ranks_%d.txt' % (outDir, outDir, inputNumber))
    #plt.show()
    #inspect_sym_input()
    if maxIndex != labelMatrix[inputNumber]:
        print "Error, correct label is", labelMatrix[inputNumber]
    return maxIndex
    
def do_experiment(inputsFile, weightsFile, metaFile, numberOfImages, outputFile):
    read_weights_from_saved_tf_model(metaFile)
    read_inputs_from_file(inputsFile, 28, 28, False)
    with open(outputFile, "w") as f:
        for i in range(numberOfImages):
            init(inputsFile, weightsFile, 28, 28)
            inputIndex = randint(0, len(inputMatrix))
            
            kerasResult = do_all_layers_keras(inputIndex)
            kerasSymOut = symInput[0,0,kerasResult]
            #Top 20% of pixels
            '''plt.figure()
            plt.imshow(get_top_pixels(kerasSymOut, 0.2))
            plt.savefig('./result_images/mnist_deep/Top 20%% Images/50_inputs_test/mnist_deep_top_20%%_sym_coeffs_%d'%i)
            plt.figure()
            plt.imshow(get_top_pixels(abs(kerasSymOut), 0.2))
            plt.savefig('./result_images/mnist_deep/Top 20%% Images/50_inputs_test/mnist_deep_top_20%%_abs_sym_coeffs_%d'%i)
            plt.figure()
            plt.imshow(get_top_pixels(np.multiply(kerasSymOut, inputMatrix[inputIndex][:,:,0]), 0.2))
            plt.savefig('./result_images/mnist_deep/Top 20%% Images/50_inputs_test/mnist_deep_top_20%%_sym_coeffs_times_in_%d'%i)'''
            #Above-average pixels
            '''plt.figure()
            plt.imshow(get_above_average_pixels(kerasSymOut))
            plt.savefig('./result_images/mnist_deep/Above_average_images/50_inputs_test/mnist_deep_above_average_sym_coeffs_%d' % i)
            plt.figure()
            plt.imshow(get_above_average_pixels(abs(kerasSymOut)))
            plt.savefig('./result_images/mnist_deep/Above_average_images/50_inputs_test/mnist_deep_above_average_abs_sym_coeffs_%d' % i)
            plt.figure()
            plt.imshow(get_above_average_pixels(np.multiply(kerasSymOut, inputMatrix[inputIndex][:,:,0])))
            plt.savefig('./result_images/mnist_deep/Above_average_images/50_inputs_test/mnist_deep_above_average_sym_coeffs_times_in_%d' % i)'''
            
            init(inputsFile, weightsFile, 28, 28)
            reluResult = do_all_layers(inputIndex, 0, 1)
            reluSymOut = symInput[0,0,reluResult]
            #Top 20% of pixels
            '''plt.figure()
            plt.imshow(get_top_pixels(reluSymOut, 0.2))
            plt.savefig('./result_images/Converted Relu Network/Top 20%% Images/50_inputs_test/Converted_relu_top_20%%_sym_coeffs_%d'%i)
            plt.figure()
            plt.imshow(get_top_pixels(abs(reluSymOut), 0.2))
            plt.savefig('./result_images/Converted Relu Network/Top 20%% Images/50_inputs_test/Converted_relu_top_20%%_abs_sym_coeffs_%d'%i)
            plt.figure()
            plt.imshow(get_top_pixels(np.multiply(reluSymOut, inputMatrix[inputIndex][:,:,0]), 0.2))
            plt.savefig('./result_images/Converted Relu Network/Top 20%% Images/50_inputs_test/Converted_relu_top_20%%_sym_coeffs_times_in_%d'%i)'''
            #Above-average pixels
            '''plt.figure()
            plt.imshow(get_above_average_pixels(reluSymOut))
            plt.savefig('./result_images/Converted Relu Network/Above_average_images/50_inputs_test/Converted_relu_above_average_sym_coeffs_%d' % i)
            plt.figure()
            plt.imshow(get_above_average_pixels(abs(reluSymOut)))
            plt.savefig('./result_images/Converted Relu Network/Above_average_images/50_inputs_test/Converted_relu_above_average_abs_sym_coeffs_%d' % i)
            plt.figure()
            plt.imshow(get_above_average_pixels(np.multiply(reluSymOut, inputMatrix[inputIndex][:,:,0])))
            plt.savefig('./result_images/Converted Relu Network/Above_average_images/50_inputs_test/Converted_relu_above_average_sym_coeffs_times_in_%d' % i)'''
            
            
            if kerasResult != reluResult or kerasResult != labelMatrix[inputIndex]:
                f.write("Houston, we have a problem: ")
                if kerasResult != reluResult:
                    f.write("Outputs don't match: %d (keras), %d\n"%(kerasResult, reluResult))
                else:
                    f.write("Keras: %d, actual: %d\n"%(kerasResult, labelMatrix[inputIndex]))
            x = compare_pixel_ranks(kerasSymOut, reluSymOut)
            y = compare_pixel_ranks(abs(kerasSymOut), abs(reluSymOut))
            z = compare_pixel_ranks(np.multiply(kerasSymOut, inputMatrix[inputIndex][:,:,0]), np.multiply(reluSymOut, inputMatrix[inputIndex][:,:,0]))
            f.write("%d %d %d\n"%(x,y,z))

def manhattan_distance(image0, image1):
    total = 0
    image0flat = image0.flatten()
    image1flat = image1.flatten()
    for i in range(len(image0)):
        total += abs(image0flat[i] - image1flat[i])
    return total
    
def euclidean_distance(x, y):
    total = 0
    x_flat = x.flatten()
    y_flat = y.flatten()
    for i in range(len(x_flat)):
        total += (x_flat[i] - y_flat[i]) ** 2
    return np.sqrt(total)

def find_closest_input_with_different_label(inputsFile, metaFile, inputIndex=-1, ckpoint='./tf_models'):
    '''Generates differential analyses, either on a random input chosen from inputsFile if inputIndex == -1, or on a file from exampleInputMatrix determined by inputIndex.'''
    read_weights_from_saved_tf_model(metaFile, ckpoint=ckpoint)
    read_inputs_from_file(inputsFile, 28, 28, False)
    
    inputImage = None
    correctLabel = -1
    if inputIndex == -1:
        inputIndex = randint(0, len(inputMatrix))
        inputImage = inputMatrix[inputIndex]
        correctLabel = labelMatrix[inputIndex]
    else:
        inputImage = exampleInputMatrix[inputIndex]
        correctLabel = inputIndex
    print "Our image is a", correctLabel
    closestImageIndex = None
    minDistance = 255*inputImage.shape[0]*inputImage.shape[1]
    
    graph = tf.Graph()
    with tf.Session() as sess:
        imported_graph = tf.train.import_meta_graph(metaFile)
        imported_graph.restore(sess, tf.train.latest_checkpoint(ckpoint))
        graph = tf.get_default_graph()
        
        x = graph.get_tensor_by_name("import/x:0")
        #keep_prob = graph.get_tensor_by_name("import/keep_prob:0")
        gradients = graph.get_tensor_by_name("import/gradients:0")
        im_data = np.array(normalize_to_1(inputImage[:,:,0]), dtype=np.float32)
        data = np.ndarray.flatten(im_data)
        #feed_dict = {x:[data], keep_prob: 1.0}
        feed_dict = {x:[data]} #tf_relu version
        input_result = gradients.eval(feed_dict=feed_dict)
        #print input_result
        for i in range(len(inputMatrix)):
            if labelMatrix[i] == correctLabel:
                continue
            im_data = normalize_to_1(inputMatrix[i][:,:,0])
            data = np.ndarray.flatten(im_data)
            #feed_dict = {x:[data], keep_prob:1.0}
            feed_dict = {x:[data]} #tf_relu version
            image_result = gradients.eval(feed_dict=feed_dict)
            #print image_result
            distance = euclidean_distance(input_result, image_result) #compare gradients
            #distance = manhattan_distance(inputImage, inputMatrix[i]) #compare @ input layer
            if distance < minDistance:
                minDistance = distance
                closestImageIndex = i
            if distance == minDistance:
                print "This image has the same distance as our current closest"
    print "Our closest image is a", labelMatrix[closestImageIndex]
    print "It has a distance of", minDistance
    print "Closest image index:", closestImageIndex
    '''plt.figure()
    plt.imshow(inputImage[:,:,0])
    plt.show()'''
    plt.figure()
    plt.imshow(inputMatrix[closestImageIndex][:,:,0])
    plt.savefig('./result_images/Differential_attributions/tf_relu_network/closest_gradients_images/%d\'s_closest_image' % correctLabel) # Just for reference
    plt.close()
    
    #read_weights_from_saved_tf_model(metaFile)
    
    init_symInput(inputImage.shape[0],inputImage.shape[1])
    inputResult = do_all_layers_keras_for_image(inputImage)
    if inputResult != correctLabel:
        print "Error: incorrect prediction, correct label is", correctLabel
        #return -1
        inputSymOut = symInput[0,0,correctLabel]
    else:
        inputSymOut = symInput[0,0,inputResult]
    
    #These are repeats from other experiments, but it helps to have them handy.
    '''plt.figure()
    plt.imshow(normalize_to_255(inputSymOut))
    plt.savefig('./result_images/Differential_attributions/%d\'s_coeffs' % correctLabel)
    #plt.imshow(get_top_pixels(inputSymOut, 0.2))
    #plt.savefig('./result_images/Differential_attributions/%d\'s_coeffs_top_20%%' % correctLabel)
    plt.imshow(image_based_on_pixel_ranks(inputSymOut))
    plt.savefig('./result_images/Differential_attributions/%d\'s_ranked_sym_coeffs' % correctLabel)
    plt.close()'''
    
    init_symInput(inputImage.shape[0],inputImage.shape[1])
    closestResult = do_all_layers_keras_for_image(inputMatrix[closestImageIndex])
    if closestResult != labelMatrix[closestImageIndex]:
        print "Error: incorrect prediction, correct label is", labelMatrix[closestImageIndex]
        #return -1
        closestSymOut = symInput[0,0,labelMatrix[closestImageIndex]]
    else:
        closestSymOut = symInput[0,0,closestResult]
    
    #The basic metrics for the closest image. Again, just to have them handy.
    '''plt.figure()
    plt.imshow(normalize_to_255(closestSymOut))
    plt.savefig('./result_images/Differential_attributions/%d\'s_closest_image_coeffs' % correctLabel)
    #plt.imshow(get_top_pixels(closestSymOut, 0.2))
    #plt.savefig('./result_images/Differential_attributions/%d\'s_closest_image_coeffs_top_20%%' % correctLabel)
    plt.imshow(image_based_on_pixel_ranks(closestSymOut))
    plt.savefig('./result_images/Differential_attributions/%d\'s_closest_image_coeffs_ranked' % correctLabel)
    write_pixel_ranks_to_file(closestSymOut, './result_images/Differential_attributions/%d\'s_closest_image_coeffs_ranks.txt' % correctLabel)'''
    #plt.close()
    #plt.show()
    
    #The actual differential analyses. First, the difference between the two sets of coeffs.
    symDistance = euclidean_distance(inputSymOut, closestSymOut)
    print "Distance between the two sets of coeffs:", symDistance
    plt.figure()
    plt.imshow(normalize_to_255(get_most_different_pixels(inputSymOut, closestSymOut)))
    plt.savefig('./result_images/Differential_attributions/tf_relu_network/difference_between_coeffs/%d_vs_%d_different_coeffs' % (correctLabel, labelMatrix[closestImageIndex]))
    plt.imshow(image_based_on_pixel_ranks(get_most_different_pixels(inputSymOut, closestSymOut)))
    write_image_to_file(get_most_different_pixels(inputSymOut, closestSymOut), './result_images/Differential_attributions/tf_relu_network/difference_between_coeffs/%d_vs_%d_different_coeffs.txt' % (correctLabel, labelMatrix[closestImageIndex]))
    plt.savefig('./result_images/Differential_attributions/tf_relu_network/difference_between_coeffs/Pixel_ranks/%d_vs_%d_different_coeffs_ranked' % (correctLabel, labelMatrix[closestImageIndex]))
    plt.close()
    write_pixel_ranks_to_file(get_most_different_pixels(inputSymOut, closestSymOut), './result_images/Differential_attributions/tf_relu_network/difference_between_coeffs/Pixel_ranks/%d_vs_%d_different_coeffs_ranks.txt' % (correctLabel, labelMatrix[closestImageIndex]))
    
    #Next, difference between the two sets of coeffs times the input. 
    plt.figure()
    plt.imshow(normalize_to_255(np.multiply(get_most_different_pixels(inputSymOut, closestSymOut), inputImage[:,:,0])))
    plt.savefig('./result_images/Differential_attributions/tf_relu_network/difference_times_input/%d_vs_%d_different_coeffs_times_in' % (correctLabel, labelMatrix[closestImageIndex]))
    plt.imshow(image_based_on_pixel_ranks(np.multiply(get_most_different_pixels(inputSymOut, closestSymOut), inputImage[:,:,0])))
    write_image_to_file(np.multiply(get_most_different_pixels(inputSymOut, closestSymOut), inputImage[:,:,0]), './result_images/Differential_attributions/tf_relu_network/difference_times_input/%d_vs_%d_different_coeffs_times_in.txt' % (correctLabel, labelMatrix[closestImageIndex]))
    plt.savefig('./result_images/Differential_attributions/tf_relu_network/difference_times_input/Pixel_ranks/%d_vs_%d_different_coeffs_times_in_ranked' % (correctLabel, labelMatrix[closestImageIndex]))
    write_pixel_ranks_to_file(np.multiply(get_most_different_pixels(inputSymOut, closestSymOut), inputImage[:,:,0]), './result_images/Differential_attributions/tf_relu_network/difference_times_input/Pixel_ranks/%d_vs_%d_different_coeffs_times_in_ranks.txt' % (correctLabel, labelMatrix[closestImageIndex]))
    plt.close()
    #plt.show()
    return closestImageIndex
    
def random_distances_experiment(inputFile, metaFile, inputIndex=-1):
    read_weights_from_saved_tf_model(metaFile)
    read_inputs_from_file(inputsFile, 28, 28, False)
    inputImage = None
    correctLabel = -1
    if inputIndex == -1:
        inputIndex = randint(0, len(inputMatrix))
        inputImage = inputMatrix[inputIndex]
        correctLabel = labelMatrix[inputIndex]
    else:
        inputImage = exampleInputMatrix[inputIndex]
        correctLabel = inputIndex
    closestImageIndex = None
    minDistance = 255*inputImage.shape[0]*inputImage.shape[1]
    for i in range(len(inputMatrix)):
        if labelMatrix[i] == correctLabel:
            continue
        #distance = euclidean_distance(inputImage, inputMatrix[i])
        distance = manhattan_distance(inputImage, inputMatrix[i])
        if distance < minDistance:
            minDistance = distance
            closestImageIndex = i
    
    randomImageIndices = np.full(10, -1)
    while -1 in randomImageIndices:
        randomImage = randint(0, len(inputMatrix))
        if randomImageIndices[labelMatrix[randomImage]] == -1:
            randomImageIndices[labelMatrix[randomImage]] = randomImage
    print "Indices of random images:", randomImageIndices        
    
    init_symInput(inputImage.shape[0],inputImage.shape[1])
    inputResult = do_all_layers_keras_for_image(inputImage)
    inputSymOut = symInput[0,0,inputResult]
    init_symInput(inputImage.shape[0],inputImage.shape[1])
    closestResult = do_all_layers_keras_for_image(inputMatrix[closestImageIndex])
    closestSymOut = symInput[0,0,closestResult]
    closestImageCoeffDistance = manhattan_distance(inputSymOut, closestSymOut)
    
    imageDistances = np.full(10, -1)
    coeffDistances = np.full(10, -1)
    for i in range(len(randomImageIndices)):
        if i == correctLabel:
            continue
        imageDistances[i] = manhattan_distance(inputImage, inputMatrix[randomImageIndices[i]])
        init_symInput(inputImage.shape[0],inputImage.shape[1])
        randomImageResult = do_all_layers_keras_for_image(inputMatrix[randomImageIndices[i]])
        randomImageSymOut = symInput[0,0,randomImageResult]
        coeffDistances[i] = manhattan_distance(inputSymOut, randomImageSymOut)

    print "Our image is a", correctLabel
    print "Our closest image is a", labelMatrix[closestImageIndex]
    print "It has a distance of", minDistance
    print "Closest image coeff distance:", closestImageCoeffDistance
    print "Random image distances:", imageDistances
    print "Random image coeff distances:", coeffDistances
    
def sufficient_distance_experiment(inputFile, metaFile, inputIndex=-1):
    read_weights_from_saved_tf_model(metaFile)
    read_inputs_from_file(inputsFile, 28, 28, False)
    inputImage = None
    correctLabel = -1
    if inputIndex == -1:
        inputIndex = randint(0, len(inputMatrix))
        inputImage = inputMatrix[inputIndex]
        correctLabel = labelMatrix[inputIndex]
    else:
        inputImage = exampleInputMatrix[inputIndex]
        correctLabel = inputIndex
    imageDistances = np.full(10, 255*inputImage.shape[0]*inputImage.shape[1])
    imageIndices = np.full(10, -1)
    for i in range(len(inputMatrix)):
        if labelMatrix[i] == correctLabel:
            continue
        #distance = euclidean_distance(inputImage, inputMatrix[i])
        distance = manhattan_distance(inputImage, inputMatrix[i])
        
        if distance < imageDistances[labelMatrix[i]]:
            imageDistances[labelMatrix[i]] = distance
            imageIndices[labelMatrix[i]] = i
    
    init_symInput(inputImage.shape[0],inputImage.shape[1])
    inputResult = do_all_layers_keras_for_image(inputImage)
    if inputResult != correctLabel:
        print "Input incorrectly classified, problem"
        return
    inputSymOut = symInput[0,0,inputResult]
    
    coeffDistances = np.full(10, -1)
    for i in range(len(imageIndices)):
        if i == correctLabel:
            continue
        init_symInput(inputImage.shape[0],inputImage.shape[1])
        imageResult = do_all_layers_keras_for_image(inputMatrix[imageIndices[i]])
        imageSymOut = symInput[0,0,imageResult]
        coeffDistances[i] = euclidean_distance(inputSymOut, imageSymOut)

    print "Our image is a", correctLabel
    imageDistances[correctLabel] = np.amin(imageDistances)+1
    print "Our closest image is a", np.argmin(imageDistances)
    print "It has a distance of", np.amin(imageDistances)
    print "Our farthest type of image is a", np.argmax(imageDistances)
    print "It has a distance of", np.amax(imageDistances)
    imageDistances[correctLabel] = -1
    print imageDistances
    coeffDistances[correctLabel] = np.amax(coeffDistances)-1
    print "Our closest coeffs are for", np.argmin(coeffDistances)
    print "Our farthest coeffs are for", np.argmax(coeffDistances)
    coeffDistances[correctLabel] = -1
    print coeffDistances
    
def get_percentage_same_ranks(gradientsFile, experimentFile):
    with open(gradientsFile) as igF:
        with open(experimentFile) as expF:
            gradientLines = igF.readlines()
            gradientRanks = np.arange(0)
            for l in range(len(gradientLines)):
                gradientRanks = np.append(gradientRanks, [int(stringIn) for stringIn in gradientLines[l].split('\t')[:-1]])
            experimentLines = expF.readlines()
            experimentRanks = np.arange(0)
            for l in range(len(experimentLines)):
                experimentRanks = np.append(experimentRanks, [int(stringIn) for stringIn in experimentLines[l].split('\t')[:-1]])
            sameRanks = compare_pixel_ranks(gradientRanks, experimentRanks, 2)
            print float(sameRanks)/len(experimentRanks)*100, "%% match"
            return float(sameRanks)/len(experimentRanks)*100
            
def get_rank_distance_from_files(file1, file2):
    with open(file1) as f1:
        with open(file2) as f2:
            f1Lines = f1.readlines()
            f1Ranks = np.arange(0)
            for l in range(len(f1Lines)):
                f1Ranks = np.append(f1Ranks, [int(stringIn) for stringIn in f1Lines[l].split('\t')[:-1]])
            f2Lines = f2.readlines()
            f2Ranks = np.arange(0)
            for l in range(len(f2Lines)):
                f2Ranks = np.append(f2Ranks, [int(stringIn) for stringIn in f2Lines[l].split('\t')[:-1]])
            distance = manhattan_distance(f1Ranks, f2Ranks)
            return distance

def generate_alex_net_mnist_gradients():
    read_weights_from_h5_file('./mnist_complicated.h5')
    parse_architecture_and_hyperparams("./model.json")
    
    graph = tf.Graph()
    with tf.Session() as sess:
        #imported_graph = tf.train.import_meta_graph('tf_models/gradients_testing.meta')
        imported_graph = tf.train.import_meta_graph('tf_models_alex/mnist_alex.meta')
        imported_graph.restore(sess, tf.train.latest_checkpoint('./tf_models_alex'))
        graph = tf.get_default_graph()
        convLayer = 0
        denseLayer = 0
        

        x = graph.get_tensor_by_name("import/x:0")
        W_conv1 = graph.get_tensor_by_name("import/W_conv1:0")
        b_conv1 = graph.get_tensor_by_name("import/b_conv1:0")
        W_conv2 = graph.get_tensor_by_name("import/W_conv2:0")
        b_conv2 = graph.get_tensor_by_name("import/b_conv2:0")
        W_conv3 = graph.get_tensor_by_name("import/W_conv3:0")
        b_conv3 = graph.get_tensor_by_name("import/b_conv3:0")
        W_conv4 = graph.get_tensor_by_name("import/W_conv4:0")
        b_conv4 = graph.get_tensor_by_name("import/b_conv4:0")
        W_fc1 = graph.get_tensor_by_name("import/W_fc1:0")
        b_fc1 = graph.get_tensor_by_name("import/b_fc1:0")
        W_fc2 = graph.get_tensor_by_name("import/W_fc2:0")
        b_fc2 = graph.get_tensor_by_name("import/b_fc2:0")
        W_fc3 = graph.get_tensor_by_name("import/W_fc3:0")
        b_fc3 = graph.get_tensor_by_name("import/b_fc3:0")
        gradients_pre_softmax = graph.get_tensor_by_name("import/gradients_pre_softmax:0")
        gradients = graph.get_tensor_by_name("import/gradients:0")
        prediction = graph.get_tensor_by_name("import/prediction:0")
        prediction2 = graph.get_tensor_by_name("import/prediction2:0")
        explanations = graph.get_tensor_by_name("import/explanations:0")
        f = open("./example_10.txt", 'r')
        lines = f.readlines()
        
        np.set_printoptions(threshold=np.nan)
        f = open("./example_10.txt", 'r')
        lines = f.readlines()
        for i in range(10):
            thing = str.split(lines[i],',')
            thing = [float(a)+0.5 for a in thing]
            #print len(thing)
            im_data = np.array(thing[1:], dtype=np.float32)
            data = np.ndarray.flatten(im_data)
            feed_dict = {x:[data], W_conv1: convWeightMatrix[0], b_conv1: convBiasMatrix[0], W_conv2: convWeightMatrix[1], b_conv2: convBiasMatrix[1], W_conv3: convWeightMatrix[2], b_conv3: convBiasMatrix[2], W_conv4: convWeightMatrix[3], b_conv4: convBiasMatrix[3], W_fc1: denseWeightMatrix[0], b_fc1: denseBiasMatrix[0], W_fc2: denseWeightMatrix[1], b_fc2: denseBiasMatrix[1], W_fc3: denseWeightMatrix[2], b_fc3: denseBiasMatrix[2]}
            #result = h_conv1.eval(feed_dict)
            base_result = gradients_pre_softmax.eval(feed_dict)
            #result1 = get_top_pixels(base_result, 0.2)
            result1 = base_result.reshape(28, 28)
            plt.figure()
            plt.imshow(normalize_to_255(result1))
            plt.savefig('./result_images/gradient_attributions/mnist_alex/gradients/gradient_test_pre_softmax_%d'%i)
            write_image_to_file(result1, './result_images/gradient_attributions/mnist_alex/gradients/gradient_test_pre_softmax_%d.txt'%i)
            write_pixel_ranks_to_file(result1, './result_images/gradient_attributions/mnist_alex/gradients/Pixel_ranks/gradient_test_pre_softmax_ranks_%d.txt' % i)
            result2 = np.multiply(base_result, data)
            result2 = result2.reshape(28, 28)
            plt.figure()
            plt.imshow(normalize_to_255(result2))
            plt.savefig('./result_images/gradient_attributions/mnist_alex/gradients_times_input/gradient_test_pre_softmax_mult_input_%d'%i)
            write_image_to_file(result2, './result_images/gradient_attributions/mnist_alex/gradients_times_input/gradient_test_pre_softmax_mult_input_%d.txt'%i)
            write_pixel_ranks_to_file(result2, './result_images/gradient_attributions/mnist_alex/gradients_times_input/Pixel_ranks/gradient_test_pre_softmax_mult_input_ranks_%d.txt' % i)
            
            base_result = gradients.eval(feed_dict)
            #result1 = get_top_pixels(base_result, 0.2)
            result1 = base_result.reshape(28, 28)
            plt.figure()
            plt.imshow(normalize_to_255(result1))
            plt.savefig('./result_images/gradient_attributions/mnist_alex/gradients/gradient_test_%d'%i)            
            write_image_to_file_scientific(result1, './result_images/gradient_attributions/mnist_alex/gradients/gradient_test_%d.txt'%i)
            write_pixel_ranks_to_file(result1, './result_images/gradient_attributions/mnist_alex/gradients/Pixel_ranks/gradient_test_ranks_%d.txt'%i)
            result2 = np.multiply(base_result, data)
            result2 = result2.reshape(28, 28)
            plt.figure()
            plt.imshow(normalize_to_255(result2))
            plt.savefig('./result_images/gradient_attributions/mnist_alex/gradients_times_input/gradient_test_mult_input_%d'%i)
            write_image_to_file_scientific(result2, './result_images/gradient_attributions/mnist_alex/gradients_times_input/gradient_test_mult_input_%d.txt'%i)
            write_pixel_ranks_to_file(result2, './result_images/gradient_attributions/mnist_alex/gradients_times_input/Pixel_ranks/gradient_test_mult_input_ranks_%d.txt'%i)
            plt.close()
            
            result1 = tf.argmax((prediction.eval(feed_dict)[0]),0)
            print('Predicted Label:')
            print(result1.eval())

            result = prediction2.eval(feed_dict)
            print('IG Prediction:')
            print(result)

            result = prediction2.eval(feed_dict)[:,result1.eval()]
            print('IG Prediction Label:')
            print(result)

            result = (explanations[result1.eval()]).eval(feed_dict)
            #print('IG Attribution:')
            #print(result)
            plt.imshow(normalize_to_255(result.reshape((28,28))))
            plt.savefig('./result_images/integrated_gradients/mnist_alex/integrated_gradients_%d' % i)
            write_image_to_file(result.reshape((28,28)), './result_images/integrated_gradients/mnist_alex/integrated_gradients_%d.txt' % i)
            write_pixel_ranks_to_file(result.reshape(28,28), './result_images/integrated_gradients/mnist_alex/Pixel_ranks/integrated_gradients_ranks_%d.txt' % i)

def generate_alex_net_cifar_gradients(inputIndex=-1):
    read_cifar_inputs("./cifar-10-batches-py/test_batch")
    read_weights_from_h5_file(cifarH5File)
    parse_architecture_and_hyperparams(cifarModelFile)
    graph = tf.Graph()
    with tf.Session() as sess:
        imported_graph = tf.train.import_meta_graph('tf_models_cifar_alex/cifar_alex.meta')
        imported_graph.restore(sess, tf.train.latest_checkpoint('./tf_models_cifar_alex'))
        graph = tf.get_default_graph()
        convLayer = 0
        denseLayer = 0
        

        x = graph.get_tensor_by_name("import/x:0")
        W_conv1 = graph.get_tensor_by_name("import/W_conv1:0")
        b_conv1 = graph.get_tensor_by_name("import/b_conv1:0")
        W_conv2 = graph.get_tensor_by_name("import/W_conv2:0")
        b_conv2 = graph.get_tensor_by_name("import/b_conv2:0")
        W_conv3 = graph.get_tensor_by_name("import/W_conv3:0")
        b_conv3 = graph.get_tensor_by_name("import/b_conv3:0")
        W_conv4 = graph.get_tensor_by_name("import/W_conv4:0")
        b_conv4 = graph.get_tensor_by_name("import/b_conv4:0")
        W_fc1 = graph.get_tensor_by_name("import/W_fc1:0")
        b_fc1 = graph.get_tensor_by_name("import/b_fc1:0")
        W_fc2 = graph.get_tensor_by_name("import/W_fc2:0")
        b_fc2 = graph.get_tensor_by_name("import/b_fc2:0")
        gradients_pre_softmax = graph.get_tensor_by_name("import/gradients_pre_softmax:0")
        gradients = graph.get_tensor_by_name("import/gradients:0")
        prediction = graph.get_tensor_by_name("import/prediction:0")
        prediction2 = graph.get_tensor_by_name("import/prediction2:0")
        explanations = graph.get_tensor_by_name("import/explanations:0")
        
        im_data = inputMatrix[inputIndex]
        data = np.ndarray.flatten(im_data)
        feed_dict = {x:[data], W_conv1: convWeightMatrix[0], b_conv1: convBiasMatrix[0], W_conv2: convWeightMatrix[1], b_conv2: convBiasMatrix[1], W_conv3: convWeightMatrix[2], b_conv3: convBiasMatrix[2], W_conv4: convWeightMatrix[3], b_conv4: convBiasMatrix[3], W_fc1: denseWeightMatrix[0], b_fc1: denseBiasMatrix[0], W_fc2: denseWeightMatrix[1], b_fc2: denseBiasMatrix[1]}
        
        base_result = gradients_pre_softmax.eval(feed_dict)
        result1 = base_result.reshape(32, 32, 3)
        plt.figure()
        #resultImage = np.sum(inputMatrix[inputIndex], axis=2)
        plt.imshow(inputMatrix[inputIndex])
        plt.savefig('./cifar_images/cifar%d' % inputIndex)
        #plt.show()
        plt.figure()
        plt.imshow(result1)
        plt.savefig('./result_images/gradient_attributions/cifar_alex/gradients/cifar_alex_gradients_pre_softmax_%d' % inputIndex)
        write_pixel_ranks_to_file(result1, './result_images/gradient_attributions/cifar_alex/gradients/Pixel_ranks/cifar_alex_gradients_pre_softmax_ranks_%d.txt' % inputIndex)
        plt.imshow(np.sum(result1, axis=2))
        plt.savefig('./result_images/gradient_attributions/cifar_alex/gradients/cifar_alex_gradients_pre_softmax_accumulated_%d' % inputIndex)
        #plt.show()
        plt.close()
        
        img = visualize_attrs_windowing(inputMatrix[inputIndex], result1)
        img.save('./result_images/gradient_attributions/cifar_alex/gradients_times_input/cifar_alex_gradients_pre_softmax_mult_input_%d.png' % inputIndex)
        #Below is much the same as what visualize_attrs_windowing does, but we need the array, not the PIL image object, to determine the ranks. 
        attrs = gray_scale(result1)
        attrs = abs(attrs)
        attrs = np.clip(attrs/np.percentile(attrs, 99), 0,1)
        vis = inputMatrixp[inputIndex]*attrs
        write_pixel_ranks_to_file(vis, './result_images/gradient_attributions/cifar_alex/gradients_times_input/Pixel_ranks/cifar_alex_gradients_pre_softmax_mult_input_ranks_%d.txt' % inputIndex)
        
        im_data = normalize_to_1(inputMatrix[inputIndex])
        data = np.ndarray.flatten(im_data)
        feed_dict2 = {x:[data], W_conv1: convWeightMatrix[0], b_conv1: convBiasMatrix[0], W_conv2: convWeightMatrix[1], b_conv2: convBiasMatrix[1], W_conv3: convWeightMatrix[2], b_conv3: convBiasMatrix[2], W_conv4: convWeightMatrix[3], b_conv4: convBiasMatrix[3], W_fc1: denseWeightMatrix[0], b_fc1: denseBiasMatrix[0], W_fc2: denseWeightMatrix[1], b_fc2: denseBiasMatrix[1]}
        base_result2 = gradients.eval(feed_dict2)
        print base_result2
        result2 = base_result2.reshape(32, 32, 3)
        plt.figure()
        plt.imshow(result2)
        plt.savefig('./result_images/gradient_attributions/cifar_alex/gradients/cifar_alex_gradients_%d' % inputIndex)
        write_pixel_ranks_to_file(result2, './result_images/gradient_attributions/cifar_alex/gradients/Pixel_ranks/cifar_alex_gradients_ranks_%d.txt' % inputIndex)
        plt.imshow(np.sum(result2, axis=2))
        plt.savefig('./result_images/gradient_attributions/cifar_alex/gradients/cifar_alex_gradients_accumulated_%d' % inputIndex)
        plt.close()
        img = visualize_attrs_windowing(inputMatrix[inputIndex], result2)
        img.save('./result_images/gradient_attributions/cifar_alex/gradients_times_input/cifar_alex_gradients_mult_input_%d.png' % inputIndex)
        attrs = gray_scale(result2)
        attrs = abs(attrs)
        attrs = np.clip(attrs/np.percentile(attrs, 99), 0,1)
        vis = inputMatrix[inputIndex]*attrs
        write_pixel_ranks_to_file(vis, './result_images/gradient_attributions/cifar_alex/gradients_times_input/Pixel_ranks/cifar_alex_gradients_mult_input_ranks_%d.txt' % inputIndex)
        
        result3 = tf.argmax((prediction.eval(feed_dict)[0]),0)
        print('Predicted Label:')
        print(result3.eval())

        result = prediction2.eval(feed_dict)
        print('IG Prediction:')
        print(result)

        result = prediction2.eval(feed_dict)[:,result3.eval()]
        print('IG Prediction Label:')
        print(result)

        result = (explanations[result3.eval()]).eval(feed_dict)
        #print('IG Attribution:')
        #print(result)
        
        img = visualize_attrs_windowing(inputMatrix[inputIndex], result.reshape(32,32,3))
        img.save('./result_images/integrated_gradients/cifar_alex/integrated_gradients_%d.png' % inputIndex)
        attrs = gray_scale(result.reshape(32,32,3))
        attrs = abs(attrs)
        attrs = np.clip(attrs/np.percentile(attrs, 99), 0,1)
        vis = inputMatrix[inputIndex]*attrs
        write_pixel_ranks_to_file(vis, './result_images/integrated_gradients/cifar_alex/Pixel_ranks/integrated_gradients_ranks_%d.txt' % inputIndex)
        

def generate_alex_net_mnist_differential_attributions(inputsFile, inputIndex=-1):
    read_weights_from_h5_file('./mnist_complicated.h5')
    parse_architecture_and_hyperparams("./model.json")
    read_inputs_from_file(inputsFile, 28, 28, False)
    
    inputImage = None
    correctLabel = -1
    if inputIndex == -1:
        inputIndex = randint(0, len(inputMatrix))
        inputImage = inputMatrix[inputIndex]
        correctLabel = labelMatrix[inputIndex]
    else:
        inputImage = exampleInputMatrix[inputIndex]
        correctLabel = inputIndex
    print "Our image is a", correctLabel
    closestImageIndex = None
    minDistance = 255*inputImage.shape[0]*inputImage.shape[1]
    
    graph = tf.Graph()
    with tf.Session() as sess:
        imported_graph = tf.train.import_meta_graph('tf_models_alex/mnist_alex.meta')
        imported_graph.restore(sess, tf.train.latest_checkpoint('./tf_models_alex'))
        graph = tf.get_default_graph()
        
        x = graph.get_tensor_by_name("import/x:0")
        W_conv1 = graph.get_tensor_by_name("import/W_conv1:0")
        b_conv1 = graph.get_tensor_by_name("import/b_conv1:0")
        W_conv2 = graph.get_tensor_by_name("import/W_conv2:0")
        b_conv2 = graph.get_tensor_by_name("import/b_conv2:0")
        W_conv3 = graph.get_tensor_by_name("import/W_conv3:0")
        b_conv3 = graph.get_tensor_by_name("import/b_conv3:0")
        W_conv4 = graph.get_tensor_by_name("import/W_conv4:0")
        b_conv4 = graph.get_tensor_by_name("import/b_conv4:0")
        W_fc1 = graph.get_tensor_by_name("import/W_fc1:0")
        b_fc1 = graph.get_tensor_by_name("import/b_fc1:0")
        W_fc2 = graph.get_tensor_by_name("import/W_fc2:0")
        b_fc2 = graph.get_tensor_by_name("import/b_fc2:0")
        W_fc3 = graph.get_tensor_by_name("import/W_fc3:0")
        b_fc3 = graph.get_tensor_by_name("import/b_fc3:0")
        gradients_pre_softmax = graph.get_tensor_by_name("import/gradients_pre_softmax:0")
        gradients = graph.get_tensor_by_name("import/gradients:0")
        prediction = graph.get_tensor_by_name("import/prediction:0")
        prediction2 = graph.get_tensor_by_name("import/prediction2:0")
        explanations = graph.get_tensor_by_name("import/explanations:0")
        
        im_data = np.array(normalize_to_1(inputImage[:,:,0]), dtype=np.float32)
        data = np.ndarray.flatten(im_data)
        feed_dict = {x:[data], W_conv1: convWeightMatrix[0], b_conv1: convBiasMatrix[0], W_conv2: convWeightMatrix[1], b_conv2: convBiasMatrix[1], W_conv3: convWeightMatrix[2], b_conv3: convBiasMatrix[2], W_conv4: convWeightMatrix[3], b_conv4: convBiasMatrix[3], W_fc1: denseWeightMatrix[0], b_fc1: denseBiasMatrix[0], W_fc2: denseWeightMatrix[1], b_fc2: denseBiasMatrix[1], W_fc3: denseWeightMatrix[2], b_fc3: denseBiasMatrix[2]}
        input_result = gradients.eval(feed_dict=feed_dict)
        
        for i in range(len(inputMatrix)):
            if labelMatrix[i] == correctLabel:
                continue
            im_data = normalize_to_1(inputMatrix[i][:,:,0])
            data = np.ndarray.flatten(im_data)
            feed_dict = {x:[data], W_conv1: convWeightMatrix[0], b_conv1: convBiasMatrix[0], W_conv2: convWeightMatrix[1], b_conv2: convBiasMatrix[1], W_conv3: convWeightMatrix[2], b_conv3: convBiasMatrix[2], W_conv4: convWeightMatrix[3], b_conv4: convBiasMatrix[3], W_fc1: denseWeightMatrix[0], b_fc1: denseBiasMatrix[0], W_fc2: denseWeightMatrix[1], b_fc2: denseBiasMatrix[1], W_fc3: denseWeightMatrix[2], b_fc3: denseBiasMatrix[2]}
            image_result = gradients.eval(feed_dict=feed_dict)
            distance = euclidean_distance(input_result, image_result)
            if distance < minDistance:
                minDistance = distance
                closestImageIndex = i
            if distance == minDistance:
                print "This image has the same distance as our current closest"
    print "Our closest image is a", labelMatrix[closestImageIndex]
    print "It has a distance of", minDistance
    print "Closest image index:", closestImageIndex
    
    plt.figure()
    plt.imshow(inputMatrix[closestImageIndex][:,:,0])
    plt.savefig('./result_images/differential_attributions/mnist_alex/closest_gradients_images/%d\'s_closest_image' % correctLabel) # Just for reference
    plt.close()
    
    init_symInput(inputImage.shape[0],inputImage.shape[1])
    inputResult = do_all_layers_keras_for_image(inputImage)
    if inputResult != correctLabel:
        print "Error: incorrect prediction, correct label is", correctLabel
        return -1
    inputSymOut = symInput[0,0,inputResult]
    
    init_symInput(inputImage.shape[0],inputImage.shape[1])
    closestResult = do_all_layers_keras_for_image(inputMatrix[closestImageIndex])
    if closestResult != labelMatrix[closestImageIndex]:
        print "Error: incorrect prediction, correct label is", labelMatrix[closestImageIndex]
        return -1
    closestSymOut = symInput[0,0,closestResult]
    
    #The actual differential analyses. First, the difference between the two sets of coeffs.
    symDistance = euclidean_distance(inputSymOut, closestSymOut)
    print "Distance between the two sets of coeffs:", symDistance
    plt.figure()
    plt.imshow(normalize_to_255(get_most_different_pixels(inputSymOut, closestSymOut)))
    plt.savefig('./result_images/differential_attributions/mnist_alex/difference_between_coeffs/%d_vs_%d_different_coeffs' % (correctLabel, labelMatrix[closestImageIndex]))
    plt.imshow(image_based_on_pixel_ranks(get_most_different_pixels(inputSymOut, closestSymOut)))
    write_image_to_file(get_most_different_pixels(inputSymOut, closestSymOut), './result_images/differential_attributions/mnist_alex/difference_between_coeffs/%d_vs_%d_different_coeffs.txt' % (correctLabel, labelMatrix[closestImageIndex]))
    plt.savefig('./result_images/differential_attributions/mnist_alex/difference_between_coeffs/Pixel_ranks/%d_vs_%d_different_coeffs_ranked' % (correctLabel, labelMatrix[closestImageIndex]))
    plt.close()
    write_pixel_ranks_to_file(get_most_different_pixels(inputSymOut, closestSymOut), './result_images/differential_attributions/mnist_alex/difference_between_coeffs/Pixel_ranks/%d_vs_%d_different_coeffs_ranks.txt' % (correctLabel, labelMatrix[closestImageIndex]))
    
    #Next, difference between the two sets of coeffs times the input. 
    plt.figure()
    plt.imshow(normalize_to_255(np.multiply(get_most_different_pixels(inputSymOut, closestSymOut), inputImage[:,:,0])))
    plt.savefig('./result_images/differential_attributions/mnist_alex/difference_times_input/%d_vs_%d_different_coeffs_times_in' % (correctLabel, labelMatrix[closestImageIndex]))
    plt.imshow(image_based_on_pixel_ranks(np.multiply(get_most_different_pixels(inputSymOut, closestSymOut), inputImage[:,:,0])))
    write_image_to_file(np.multiply(get_most_different_pixels(inputSymOut, closestSymOut), inputImage[:,:,0]), './result_images/differential_attributions/mnist_alex/difference_times_input/%d_vs_%d_different_coeffs_times_in.txt' % (correctLabel, labelMatrix[closestImageIndex]))
    plt.savefig('./result_images/differential_attributions/mnist_alex/difference_times_input/Pixel_ranks/%d_vs_%d_different_coeffs_times_in_ranked' % (correctLabel, labelMatrix[closestImageIndex]))
    write_pixel_ranks_to_file(np.multiply(get_most_different_pixels(inputSymOut, closestSymOut), inputImage[:,:,0]), './result_images/differential_attributions/mnist_alex/difference_times_input/Pixel_ranks/%d_vs_%d_different_coeffs_times_in_ranks.txt' % (correctLabel, labelMatrix[closestImageIndex]))
    plt.close()
    #plt.show()
    return closestImageIndex
    
def generate_alex_net_cifar_differential_attributions(inputsFile, inputIndex):
    '''Generates differential attributions for the given input file. Calls do_all_layers_keras_3d, so it also saves the coefficients and coeffs*input for both the selected image and the closest one as a side effect.'''
    read_cifar_inputs(inputsFile)
    read_weights_from_h5_file(cifarH5File)
    parse_architecture_and_hyperparams(cifarModelFile)
    
    inputImage = inputMatrix[inputIndex]
    correctLabel = labelMatrix[inputIndex]
    print "Our image is a", correctLabel
    closestImageIndex = None
    minDistance = 255*inputImage.shape[0]*inputImage.shape[1]
    
    graph = tf.Graph()
    with tf.Session() as sess:
        imported_graph = tf.train.import_meta_graph('tf_models_cifar_alex/cifar_alex.meta')
        imported_graph.restore(sess, tf.train.latest_checkpoint('./tf_models_cifar_alex'))
        graph = tf.get_default_graph()

        x = graph.get_tensor_by_name("import/x:0")
        W_conv1 = graph.get_tensor_by_name("import/W_conv1:0")
        b_conv1 = graph.get_tensor_by_name("import/b_conv1:0")
        W_conv2 = graph.get_tensor_by_name("import/W_conv2:0")
        b_conv2 = graph.get_tensor_by_name("import/b_conv2:0")
        W_conv3 = graph.get_tensor_by_name("import/W_conv3:0")
        b_conv3 = graph.get_tensor_by_name("import/b_conv3:0")
        W_conv4 = graph.get_tensor_by_name("import/W_conv4:0")
        b_conv4 = graph.get_tensor_by_name("import/b_conv4:0")
        W_fc1 = graph.get_tensor_by_name("import/W_fc1:0")
        b_fc1 = graph.get_tensor_by_name("import/b_fc1:0")
        W_fc2 = graph.get_tensor_by_name("import/W_fc2:0")
        b_fc2 = graph.get_tensor_by_name("import/b_fc2:0")
        gradients_pre_softmax = graph.get_tensor_by_name("import/gradients_pre_softmax:0")
        gradients = graph.get_tensor_by_name("import/gradients:0")
        prediction = graph.get_tensor_by_name("import/prediction:0")
        prediction2 = graph.get_tensor_by_name("import/prediction2:0")
        explanations = graph.get_tensor_by_name("import/explanations:0")
        
        im_data = normalize_to_1(inputImage)
        data = np.ndarray.flatten(im_data)
        feed_dict = {x:[data], W_conv1: convWeightMatrix[0], b_conv1: convBiasMatrix[0], W_conv2: convWeightMatrix[1], b_conv2: convBiasMatrix[1], W_conv3: convWeightMatrix[2], b_conv3: convBiasMatrix[2], W_conv4: convWeightMatrix[3], b_conv4: convBiasMatrix[3], W_fc1: denseWeightMatrix[0], b_fc1: denseBiasMatrix[0], W_fc2: denseWeightMatrix[1], b_fc2: denseBiasMatrix[1]}
        input_result = gradients.eval(feed_dict=feed_dict)
        
        for i in range(len(inputMatrix)):
            if labelMatrix[i] == correctLabel:
                continue
            im_data = normalize_to_1(inputMatrix[i])
            data = np.ndarray.flatten(im_data)
            feed_dict = {x:[data], W_conv1: convWeightMatrix[0], b_conv1: convBiasMatrix[0], W_conv2: convWeightMatrix[1], b_conv2: convBiasMatrix[1], W_conv3: convWeightMatrix[2], b_conv3: convBiasMatrix[2], W_conv4: convWeightMatrix[3], b_conv4: convBiasMatrix[3], W_fc1: denseWeightMatrix[0], b_fc1: denseBiasMatrix[0], W_fc2: denseWeightMatrix[1], b_fc2: denseBiasMatrix[1]}
            image_result = gradients.eval(feed_dict=feed_dict)
            distance = euclidean_distance(input_result, image_result)
            if distance < minDistance:
                minDistance = distance
                print distance, i
                closestImageIndex = i
            elif distance == minDistance:
                print "Image", i, "has the same distance as our current closest"
    print "Our closest image is a", labelMatrix[closestImageIndex]
    print "It has a distance of", minDistance
    print "Closest image index:", closestImageIndex
        
    plt.figure()
    plt.imshow(inputMatrix[closestImageIndex])
    plt.savefig('./result_images/differential_attributions/cifar_alex/closest_gradients_images/%d\'s_closest_image' % inputIndex) # Just for reference
    plt.close()
        
    init_3d_symInput(inputImage.shape[0],inputImage.shape[1])
    inputResult = do_all_layers_keras_3d(inputIndex, 'cifar_alex')
    if inputResult != correctLabel:
        print "Error: incorrect prediction, correct label is", correctLabel
        #return -1
    inputSymOut = symInput[0,0,inputResult]
        
    init_3d_symInput(inputImage.shape[0],inputImage.shape[1])
    closestResult = do_all_layers_keras_3d(closestImageIndex, 'cifar_alex')
    if closestResult != labelMatrix[closestImageIndex]:
        print "Error: incorrect prediction, correct label is", labelMatrix[closestImageIndex]
            #return -1
    closestSymOut = symInput[0,0,closestResult]
        
    #The actual differential analyses. First, the difference between the two sets of coeffs.
    symDistance = euclidean_distance(inputSymOut, closestSymOut)
    print "Distance between the two sets of coeffs:", symDistance
    plt.figure()
    plt.imshow(normalize_to_255(get_most_different_pixels(inputSymOut, closestSymOut)))
    plt.savefig('./result_images/differential_attributions/cifar_alex/difference_between_coeffs/%d_vs_%d_different_coeffs' % (inputIndex, closestImageIndex))
    write_image_to_file(get_most_different_pixels(inputSymOut, closestSymOut), './result_images/differential_attributions/cifar_alex/difference_between_coeffs/%d_vs_%d_different_coeffs.txt' % (inputIndex, closestImageIndex))
    write_pixel_ranks_to_file(get_most_different_pixels(inputSymOut, closestSymOut), './result_images/differential_attributions/cifar_alex/difference_between_coeffs/Pixel_ranks/%d_vs_%d_different_coeffs_ranks.txt' % (inputIndex, closestImageIndex))
    #plt.imshow(image_based_on_pixel_ranks(get_most_different_pixels(inputSymOut, closestSymOut)))
    #plt.savefig('./result_images/differential_attributions/cifar_alex/difference_between_coeffs/Pixel_ranks/%d_vs_%d_different_coeffs_ranked' % (inputIndex, labelMatrix[closestImageIndex]))
    plt.close()
        
    #Next, difference between the two sets of coeffs times the input. 
    plt.figure()
    diffTimesInput = visualize_attrs_windowing(inputImage, get_most_different_pixels(inputSymOut, closestSymOut))
    
    diffTimesInput.save('./result_images/differential_attributions/cifar_alex/difference_times_input/%d_vs_%d_different_coeffs_times_in.png' % (inputIndex, closestImageIndex))
    attrs = gray_scale(get_most_different_pixels(inputSymOut, closestSymOut))
    attrs = abs(attrs)
    attrs = np.clip(attrs/np.percentile(attrs, 99), 0,1)
    vis = inputMatrix[inputIndex]*attrs
    write_image_to_file(vis, './result_images/differential_attributions/cifar_alex/difference_times_input/%d_vs_%d_different_coeffs_times_in.txt' % (inputIndex, closestImageIndex))
    write_pixel_ranks_to_file(vis, './result_images/differential_attributions/cifar_alex/difference_times_input/Pixel_ranks/%d_vs_%d_different_coeffs_times_in_ranks.txt' % (inputIndex, closestImageIndex))
    #plt.imshow(image_based_on_pixel_ranks(np.multiply(get_most_different_pixels(inputSymOut, closestSymOut), inputImage[:,:,0])))
    #plt.savefig('./result_images/differential_attributions/cifar_alex/difference_times_input/Pixel_ranks/%d_vs_%d_different_coeffs_times_in_ranked' % (correctLabel, labelMatrix[closestImageIndex]))
    plt.close()
    #plt.show()
    return closestImageIndex
        

def generate_relu_differential_attributions(inputsFile, inputIndex=-1):
    read_weights_from_file("./mnist_3A_layer.txt")
    read_inputs_from_file(inputsFile, 28, 28, False)
    
    inputImage = None
    correctLabel = -1
    if inputIndex == -1:
        inputIndex = randint(0, len(inputMatrix))
        inputImage = inputMatrix[inputIndex]
        correctLabel = labelMatrix[inputIndex]
    else:
        inputImage = exampleInputMatrix[inputIndex]
        correctLabel = inputIndex
    print "Our image is a", correctLabel
    closestImageIndex = None
    minDistance = 255*inputImage.shape[0]*inputImage.shape[1]
    
    graph = tf.Graph()
    with tf.Session() as sess:
        imported_graph = tf.train.import_meta_graph("./tf_models_relu_framework/mnist_relu_framework.meta")
        imported_graph.restore(sess, tf.train.latest_checkpoint("./tf_models_relu_framework"))
        graph = tf.get_default_graph()
        convLayer = 0
        denseLayer = 0
        
        x = graph.get_tensor_by_name("import/x:0")
        W_conv1 = graph.get_tensor_by_name("import/W_conv1:0")
        b_conv1 = graph.get_tensor_by_name("import/b_conv1:0")
        W_conv2 = graph.get_tensor_by_name("import/W_conv2:0")
        b_conv2 = graph.get_tensor_by_name("import/b_conv2:0")
        W_conv3 = graph.get_tensor_by_name("import/W_conv3:0")
        b_conv3 = graph.get_tensor_by_name("import/b_conv3:0")
        W_conv4 = graph.get_tensor_by_name("import/W_conv4:0")
        b_conv4 = graph.get_tensor_by_name("import/b_conv4:0")
        gradients_pre_softmax = graph.get_tensor_by_name("import/gradients_pre_softmax:0")
        gradients = graph.get_tensor_by_name("import/gradients:0")
        prediction = graph.get_tensor_by_name("import/prediction:0")
        prediction2 = graph.get_tensor_by_name("import/prediction2:0")
        explanations = graph.get_tensor_by_name("import/explanations:0")
        
        im_data = np.array(normalize_to_1(inputImage[:,:,0]), dtype=np.float32)
        data = np.ndarray.flatten(im_data)
        feed_dict = {x:[data], W_conv1: weightMatrix[0], b_conv1: biasMatrix[0], W_conv2: weightMatrix[1], b_conv2: biasMatrix[1], W_conv3: weightMatrix[2], b_conv3: biasMatrix[2], W_conv4: weightMatrix[3], b_conv4: biasMatrix[3]}
        input_result = gradients.eval(feed_dict=feed_dict)
        
        for i in range(len(inputMatrix)):
            if labelMatrix[i] == correctLabel:
                continue
            im_data = normalize_to_1(inputMatrix[i][:,:,0])
            data = np.ndarray.flatten(im_data)
            feed_dict = {x:[data], W_conv1: weightMatrix[0], b_conv1: biasMatrix[0], W_conv2: weightMatrix[1], b_conv2: biasMatrix[1], W_conv3: weightMatrix[2], b_conv3: biasMatrix[2], W_conv4: weightMatrix[3], b_conv4: biasMatrix[3]}
            image_result = gradients.eval(feed_dict=feed_dict)
            distance = euclidean_distance(input_result, image_result)
            if distance < minDistance:
                minDistance = distance
                closestImageIndex = i
            if distance == minDistance:
                print "This image has the same distance as our current closest"
    print "Our closest image is a", labelMatrix[closestImageIndex]
    print "It has a distance of", minDistance
    print "Closest image index:", closestImageIndex
    
    plt.figure()
    plt.imshow(inputMatrix[closestImageIndex][:,:,0])
    plt.savefig('./result_images/differential_attributions/relu_network/closest_gradients_images/%d\'s_closest_image' % correctLabel) # Just for reference
    plt.close()
    
    read_weights_from_file("./mnist_3A_layer.txt")
    init_symInput(inputImage.shape[0],inputImage.shape[1])
    inputResult = do_all_layers_for_image(inputImage)
    if inputResult != correctLabel:
        print "Error: incorrect prediction, correct label is", correctLabel
        return -1
    inputSymOut = symInput[0,0,inputResult]
    
    read_weights_from_file("./mnist_3A_layer.txt")
    init_symInput(inputImage.shape[0],inputImage.shape[1])
    closestResult = do_all_layers_for_image(inputMatrix[closestImageIndex])
    if closestResult != labelMatrix[closestImageIndex]:
        print "Error: incorrect prediction, correct label is", labelMatrix[closestImageIndex]
        return -1
    closestSymOut = symInput[0,0,closestResult]
    
    #The actual differential analyses. First, the difference between the two sets of coeffs.
    symDistance = euclidean_distance(inputSymOut, closestSymOut)
    print "Distance between the two sets of coeffs:", symDistance
    plt.figure()
    plt.imshow(normalize_to_255(get_most_different_pixels(inputSymOut, closestSymOut)))
    plt.savefig('./result_images/differential_attributions/relu_network/difference_between_coeffs/%d_vs_%d_different_coeffs' % (correctLabel, labelMatrix[closestImageIndex]))
    plt.imshow(image_based_on_pixel_ranks(get_most_different_pixels(inputSymOut, closestSymOut)))
    write_image_to_file(get_most_different_pixels(inputSymOut, closestSymOut), './result_images/differential_attributions/relu_network/difference_between_coeffs/%d_vs_%d_different_coeffs.txt' % (correctLabel, labelMatrix[closestImageIndex]))
    plt.savefig('./result_images/differential_attributions/relu_network/difference_between_coeffs/Pixel_ranks/%d_vs_%d_different_coeffs_ranked' % (correctLabel, labelMatrix[closestImageIndex]))
    plt.close()
    write_pixel_ranks_to_file(get_most_different_pixels(inputSymOut, closestSymOut), './result_images/differential_attributions/relu_network/difference_between_coeffs/Pixel_ranks/%d_vs_%d_different_coeffs_ranks.txt' % (correctLabel, labelMatrix[closestImageIndex]))
    
    #Next, difference between the two sets of coeffs times the input. 
    plt.figure()
    plt.imshow(normalize_to_255(np.multiply(get_most_different_pixels(inputSymOut, closestSymOut), inputImage[:,:,0])))
    plt.savefig('./result_images/differential_attributions/relu_network/difference_times_input/%d_vs_%d_different_coeffs_times_in' % (correctLabel, labelMatrix[closestImageIndex]))
    plt.imshow(image_based_on_pixel_ranks(np.multiply(get_most_different_pixels(inputSymOut, closestSymOut), inputImage[:,:,0])))
    write_image_to_file(np.multiply(get_most_different_pixels(inputSymOut, closestSymOut), inputImage[:,:,0]), './result_images/differential_attributions/relu_network/difference_times_input/%d_vs_%d_different_coeffs_times_in.txt' % (correctLabel, labelMatrix[closestImageIndex]))
    plt.savefig('./result_images/differential_attributions/relu_network/difference_times_input/Pixel_ranks/%d_vs_%d_different_coeffs_times_in_ranked' % (correctLabel, labelMatrix[closestImageIndex]))
    write_pixel_ranks_to_file(np.multiply(get_most_different_pixels(inputSymOut, closestSymOut), inputImage[:,:,0]), './result_images/differential_attributions/relu_network/difference_times_input/Pixel_ranks/%d_vs_%d_different_coeffs_times_in_ranks.txt' % (correctLabel, labelMatrix[closestImageIndex]))
    plt.close()
    #plt.show()
    return closestImageIndex

def generate_relu_gradients_and_integrated_grads(inputIndex):
    read_weights_from_file("./mnist_3A_layer.txt")
    
    inputImage = exampleInputMatrix[inputIndex]
    
    graph = tf.Graph()
    with tf.Session() as sess:
        imported_graph = tf.train.import_meta_graph("./tf_models_relu_framework/mnist_relu_framework.meta")
        imported_graph.restore(sess, tf.train.latest_checkpoint("./tf_models_relu_framework"))
        graph = tf.get_default_graph()
        convLayer = 0
        denseLayer = 0
        
        x = graph.get_tensor_by_name("import/x:0")
        W_conv1 = graph.get_tensor_by_name("import/W_conv1:0")
        b_conv1 = graph.get_tensor_by_name("import/b_conv1:0")
        W_conv2 = graph.get_tensor_by_name("import/W_conv2:0")
        b_conv2 = graph.get_tensor_by_name("import/b_conv2:0")
        W_conv3 = graph.get_tensor_by_name("import/W_conv3:0")
        b_conv3 = graph.get_tensor_by_name("import/b_conv3:0")
        W_conv4 = graph.get_tensor_by_name("import/W_conv4:0")
        b_conv4 = graph.get_tensor_by_name("import/b_conv4:0")
        gradients_pre_softmax = graph.get_tensor_by_name("import/gradients_pre_softmax:0")
        gradients = graph.get_tensor_by_name("import/gradients:0")
        prediction = graph.get_tensor_by_name("import/prediction:0")
        prediction2 = graph.get_tensor_by_name("import/prediction2:0")
        explanations = graph.get_tensor_by_name("import/explanations:0")
        
        im_data = np.array(normalize_to_1(inputImage[:,:,0]), dtype=np.float32)
        data = np.ndarray.flatten(im_data)
        feed_dict = {x:[data], W_conv1: weightMatrix[0], b_conv1: biasMatrix[0], W_conv2: weightMatrix[1], b_conv2: biasMatrix[1], W_conv3: weightMatrix[2], b_conv3: biasMatrix[2], W_conv4: weightMatrix[3], b_conv4: biasMatrix[3]}
        input_result = gradients.eval(feed_dict=feed_dict)
        
        base_result = gradients_pre_softmax.eval(feed_dict)
        #result1 = get_top_pixels(base_result, 0.2)
        result1 = base_result.reshape(28, 28)
        plt.figure()
        plt.imshow(normalize_to_255(result1))
        plt.savefig('./result_images/gradient_attributions/relu_network/gradients/relu_gradients_pre_softmax_%d'% inputIndex)
        write_image_to_file(result1, './result_images/gradient_attributions/relu_network/gradients/relu_gradients_pre_softmax_%d.txt'% inputIndex)
        write_pixel_ranks_to_file(result1, './result_images/gradient_attributions/relu_network/gradients/Pixel_ranks/relu_gradients_pre_softmax_ranks_%d.txt' % inputIndex)
        result2 = np.multiply(base_result, data)
        result2 = result2.reshape(28, 28)
        plt.figure()
        plt.imshow(normalize_to_255(result2))
        plt.savefig('./result_images/gradient_attributions/relu_network/gradients_times_input/relu_gradients_pre_softmax_mult_input_%d'% inputIndex)
        write_image_to_file(result2, './result_images/gradient_attributions/relu_network/gradients_times_input/relu_gradients_pre_softmax_mult_input_%d.txt'% inputIndex)
        write_pixel_ranks_to_file(result2, './result_images/gradient_attributions/relu_network/gradients_times_input/Pixel_ranks/relu_gradients_pre_softmax_mult_input_ranks_%d.txt' % inputIndex)
        
        base_result = gradients.eval(feed_dict)
        #result1 = get_top_pixels(base_result, 0.2)
        result1 = base_result.reshape(28, 28)
        plt.figure()
        plt.imshow(normalize_to_255(result1))
        plt.savefig('./result_images/gradient_attributions/relu_network/gradients/relu_gradients_%d'% inputIndex)
        write_image_to_file_scientific(result1, './result_images/gradient_attributions/relu_network/gradients/relu_gradients_%d.txt'% inputIndex)
        write_pixel_ranks_to_file(result1, './result_images/gradient_attributions/relu_network/gradients/Pixel_ranks/relu_gradients_ranks_%d.txt'% inputIndex)
        result2 = np.multiply(base_result, data)
        result2 = result2.reshape(28, 28)
        plt.figure()
        plt.imshow(normalize_to_255(result2))
        plt.savefig('./result_images/gradient_attributions/relu_network/gradients_times_input/relu_gradients_mult_input_%d'% inputIndex)
        write_image_to_file_scientific(result2, './result_images/gradient_attributions/relu_network/gradients_times_input/relu_gradients_mult_input_%d.txt'% inputIndex)
        write_pixel_ranks_to_file(result2, './result_images/gradient_attributions/relu_network/gradients_times_input/Pixel_ranks/relu_gradients_mult_input_ranks_%d.txt'% inputIndex)
        plt.close()
        
        result1 = tf.argmax((prediction.eval(feed_dict)[0]),0)
        print('Predicted Label:')
        print(result1.eval())

        result = prediction2.eval(feed_dict)
        print('IG Prediction:')
        print(result)

        result = prediction2.eval(feed_dict)[:,result1.eval()]
        print('IG Prediction Label:')
        print(result)

        result = (explanations[result1.eval()]).eval(feed_dict)
        #print('IG Attribution:')
        #print(result)
        plt.imshow(normalize_to_255(result.reshape((28,28))))
        plt.savefig('./result_images/integrated_gradients/relu_network/integrated_gradients_%d' % inputIndex)
        write_image_to_file(result.reshape((28,28)), './result_images/integrated_gradients/relu_network/integrated_gradients_%d.txt' % inputIndex)
        write_pixel_ranks_to_file(result.reshape(28,28), './result_images/integrated_gradients/relu_network/Pixel_ranks/integrated_gradients_ranks_%d.txt' % inputIndex)

weightsFile = "./mnist_3A_layer.txt"
inputsFile = "./mnist_train.csv"
exampleInputsFile = "./example_10.txt"
cifarInputsFile = "./cifar-10-batches-py/test_batch"
h5File = "./mnist_complicated.h5"
cifarH5File = "./cifar10_complicated.h5"
cifarModelFile = "./cifar_model/model.json"
modelFile = "./model.json"
metaFile = "./tf_models/mnist.meta"
altMetaFile = './tf_models/gradients_testing_20000.meta'
noDropoutMetaFile = "./tf_models/mnist_no_dropout.meta"
noPoolingMetaFile = "./tf_models/mnist_no_pooling.meta"
reluMetaFile = "./tf_models_relu/mnist_relu_network.meta"
reluFrameworkMetaFile = "./tf_models_relu_framework/mnist_relu_framework.meta"
alexMetaFile ="./tf_models_alex/mnist_alex.meta"
gradientsTestingMetaFile = './tf.models/gradients_testing.meta'
checkpoint = "./tf_models"
reluCheckpoint = "./tf_models_relu"
reluFrameworkCheckpoint = "./tf_models_relu_framework"
alexCheckpoint = "./tf_models_alex"
gradientRanksFile = "./result_images/gradient_test/gradient_test_pre_softmax_ranks_0.txt"
experimentRanksFile = "./result_images/mnist_deep/pixel_ranks/mnist_deep_sym_coeffs_ranks_0.txt"
inputIndex = 3

# Textual Experiments
input_sentence_file = './textual_models/sentiment-analysis-keras-conv/examples.txt'
restored_sentence_file = './textual_models/sentiment-analysis-keras-conv/example_sent.txt'
sentiment_h5_file = './textual_models/sentiment-analysis-keras-conv/model-len30-embed30.h5'
sentiment_json_file = './textual_models/sentiment-analysis-keras-conv/model-len30-embed30.json'

sentence_list = list()
with open(restored_sentence_file, 'r') as f:
    line = f.readlines()[0]
    for w in line.split(',')[1:]:
	w = w.strip() #Removes leading, ending whitespaces
        sentence_list.append(w)
print(sentence_list)

# Load the weights and layers
read_weights_from_h5_file(sentiment_h5_file)
parse_architecture_and_hyperparams(sentiment_json_file)

# Read input from test dataset
read_inputs_from_file(input_sentence_file, 30, 1, plusPointFive=False)
init_symInput(30, 30)
# Running the example input concolically
# TO DO Add segment that passes the input through the embedding layer.
# TO DO Modify conv_layer_forward_ineff()
# TO DO Remove the image stuff at the end
do_all_layers_keras(0, './textual_models')


coeffxinput = np.multiply(symInput[0,0,0], embedInputMatrix[:,:,0])
print coeffxinput.shape
print coeffxinput
for (i,w) in enumerate(sentence_list):
    embed_coeff = symInput[:,:,:,i,:]
    input_val = embedInputMatrix[i,:,:]
    multiplied = coeffxinput[i,:]
    total_imp = sum(multiplied)
    print '{}th Word: {}'.format(i,w)
    print 'Multiplied summed over word:{}'.format(total_imp)
    #print 'Coefficients: {}'.format(embed_coeff)
    #print 'Word as embedding: {}'.format(input_val)
    #print 'Multiplied:{}'.format(multiplied)






#read_inputs_from_file(exampleInputsFile, 28, 28, True)
#exampleInputMatrix = np.multiply(255, inputMatrix)
#exampleInputMatrix = inputMatrix
#labelMatrix = np.arange(10)

#do_experiment(inputsFile, weightsFile, metaFile, 50, "./out.txt")

#Use this one for differential analysis of mnist_deep and tf_relu networks. Gradient analysis of mnist_deep (and tf_relu, if you really need it) is done in test.py/tf_testing_3()
#find_closest_input_with_different_label(inputsFile, reluMetaFile, inputIndex, ckpoint=reluCheckpoint)

#Use these for differential and gradient analysis of mnist_alex network.
#generate_alex_net_mnist_differential_attributions(inputsFile, inputIndex)
#generate_alex_net_mnist_gradients()

#Use these for differential and gradient analysis of our original relu network.
#generate_relu_differential_attributions(inputsFile, inputIndex)
#generate_relu_gradients_and_integrated_grads(inputIndex)

#random_distances_experiment(inputsFile, metaFile, inputIndex=0)
#sufficient_distance_experiment(inputsFile, metaFile, inputIndex=9)
#get_percentage_same_ranks(gradientRanksFile, experimentRanksFile)

#Get coefficients for cifar-10 images using alexnet. 
#read_cifar_inputs(cifarInputsFile)
#read_weights_from_h5_file(cifarH5File)
#parse_architecture_and_hyperparams(cifarModelFile)
#init_3d_symInput(32, 32)
#kerasResult = do_all_layers_keras_3d(inputIndex, 'cifar_alex')

#generate_alex_net_cifar_differential_attributions(cifarInputsFile, inputIndex)

#for i in range(10):
#generate_alex_net_cifar_gradients(i)
'''plt.imshow(inputMatrix[i])
    plt.show()
    img = PIL.Image.fromarray(inputMatrix[i])
    #img.save('./cifar_images/cifar%d.png' % i)
    img.show()'''

#Get coefficients for mnist images using (non-tf) relu network.
#init(exampleInputsFile, weightsFile, 28, 28, True)
#do_all_layers(inputIndex, 0, 1)

#Get coefficients for mnist images using tf_relu or mnist_deep networks. Read weights from correct meta file to choose between them.. 
#read_weights_from_saved_tf_model(reluMetaFile, ckpoint=relu=Checkpoint)
#init(exampleInputsFile, weightsFile, 28, 28, True)
#kerasResult = do_all_layers_keras(inputIndex, 'relu_network')

#Get coefficients for mnist images using alexnet.
#read_weights_from_h5_file(h5File)
#parse_architecture_and_hyperparams(modelFile)
#init(exampleInputsFile, weightsFile, 28, 28, True)
#kerasResult = do_all_layers_keras(inputIndex, 'mnist_alex')

#Prints distances between pairs of rank files. Can't use the loop for differential, since the names aren't consistent (0 vs 6, 1 vs 7, etc.)
#total = 0
#for i in range(len(exampleInputMatrix)):
    #d=get_rank_distance_from_files(('./result_images/inputs/Pixel_ranks/mnist_input_image_ranks_%d.txt' % inputIndex), ('./result_images/differential_attributions/mnist_alex/difference_times_input/Pixel_ranks/%d_vs_1_different_coeffs_times_in_ranks.txt'%inputIndex))
    #print d
    #total += d
#print total/10
