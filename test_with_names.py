#Just a test playground to mess around with bits and pieces before they're placed in the full-scale implementations.

import numpy as np;
#import matplotlib.pyplot as plt
import time
#import h5py
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from google.protobuf import json_format

from tensorflow.examples.tutorials.mnist import input_data

weightMatrix = None
biasMatrix = None
inputMatrix = None
symInput = None

convWeightMatrix = None
convBiasMatrix = None
denseWeightMatrix = None
denseBiasMatrix = None
layerTypeList = []
maxPoolParams = []
activationTypeList = [] 
convParams = []

#Assumed format of inputs file: list of inputs of size S+1, where the first element is discardable.
#Populates inputMatrix, a matrix of N 1-by-1-by-height-by-width-by 4D matrices, where each is an input. To make it N Sx1x1 column vecotrs, replace final nested for loops with final line. For N 1xSx1 row vectors, replace np.transpose(k) with k. Assuming single, 2D images.
def read_inputs_from_file(inputFile, height, width):
    global inputMatrix
    with open(inputFile) as f:
        lines = f.readlines()
        #print len(lines), "examples"
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
    #print "Im2Col input shape:", input_shape
    #C, H, W = input_shape
    #print "C, H, W: ",C,H,W
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1
    
    #print "Out height, width:", out_height, out_width
    
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
    #print shp
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(A_padded, shape=shp, strides=strd)
    return out_view.reshape(block_shape[0]*block_shape[1],-1)[:,::stepsize]

def conv_layer_forward(input, filter, b, stride=1, padding=1):
    #print "Beginning convolutional layer"
    #print "Conv: input shape",input.shape
    n_x, d_x, h_x, w_x = input.shape #Still assuming one 2D image at a time for now, so d_x should start as 1 and then depend on the number of filters, n_x is always 1
    #print "Conv: filter shape",filter.shape
    n_filters, d_filter, h_filter, w_filter = filter.shape # This is our usual format for this. Typical format for input is h x w x d.
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1
    input_col = im2col_indices(input, h_filter, w_filter, padding=padding, stride=stride) #im2col_sliding_strided(input, (h_filter, w_filter), stride, padding) 
    #print "Input_col shape",input_col.shape
    filter_col = filter.reshape(n_filters, -1)
    #print "Filter_col shape",filter_col.shape
    converted_biases = np.array(b).reshape(-1, 1)
    #print "10x1?",converted_biases.shape
    out = np.add(np.dot(filter_col, input_col), converted_biases)
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2) #Turns it back to (n_x, n_filters, h_out, w_out), where n_x should be 1
    #print "output shape",out.shape
    #print ""
    return out;
    
#TODO: Finish this for-loop version for individual images. No n_x here.
def conv_layer_forward_ineff(input, filters, biases, stride=1, padding=1):
    #print "Beginning conv layer"
    #print input.shape
    n_x, d_x, h_x, w_x = input.shape 
    n_filters, d_filter, h_filter, w_filter = filters.shape #d_x should equal d_filter
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1
    #print h_out, "x", w_out, "out"
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
                    #print "Total for this filter:", out[im][i][j][k]
    #print ""
    return out;
    
def relu_layer_forward(x):
    relu = lambda x: x * (x > 0).astype(float)
    return relu(x);
    
#We could make this multi-channel with an "n" parameter, changes needed noted in comments
def pool_layer_forward(X, size, stride = 1):
    #print "Beginning pooling layer"
    #print "X shape:",X.shape
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
    #print ""
    return out;
    
def pool_layer_forward_ineff(X, size, stride = 1):
    #print "X shape:",X.shape
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
def sym_conv_layer_forward(input, filters, b, stride=1, padding=1):
    #print "Beginning sym conv layer"
    h_prev, w_prev, d_prev, h_x, w_x = input.shape
    n_filters, d_filter, h_filter, w_filter = filters.shape #d_x should equal d_filter
    h_out = (h_prev - h_filter + 2 * padding) / stride + 1
    w_out = (w_prev - w_filter + 2 * padding) / stride + 1
    input_padded = np.pad(input,((0,0),(0,0),(0,0),(padding, padding),(padding, padding)),mode='constant')
    #print "Padded input shape:", input_padded.shape, "filters shape:", filters.shape
    out = np.zeros((h_out, w_out, n_filters, h_x, w_x))
    #print "Output shape", out.shape
    for i in range(n_filters):
        for j in range(0, h_out, stride):
            for k in range(0, w_out, stride):
                for l in range(d_prev):
                    #print filters[i,l].shape
                    #print input_padded[j:j+h_filter,k:k+w_filter,l].shape
                    temp = np.zeros((h_x, w_x))
                    scaledMatrices = np.multiply(filters[i,l], input_padded[j:j+h_filter,k:k+w_filter,l])
                    for m in range(h_filter):
                        for n in range(w_filter):
                            temp = np.add(temp, scaledMatrices[m,n])
                    out[j,k,i] = np.add(out[j,k,i], temp)
                #out[j,k,i] = np.add(out[j,k,i], b[i])
    #print ""
    return out

def init(inputFile, weightFile, inputHeight, inputWidth):
    #print "Initializing..."
    global symInput
    read_inputs_from_file(inputFile, inputHeight, inputWidth)
    read_weights_from_file(weightFile)
    symInput = np.zeros((inputHeight, inputWidth, 1, inputHeight, inputWidth))
    for i in range(inputHeight):
        for j in range(inputWidth):
            symInput[i,j,0,i,j] = 1
    
def classify(processedArray):
    maxValue = 0
    maxIndex = -1
    for i in range(processedArray.shape[1]):
        #print "Class",i,"confidence",processedArray[0][i][0][0]
        if(processedArray[0][i][0][0] > maxValue):
            maxValue = processedArray[0][i][0][0]
            maxIndex = i
    #print "MaxIndex:",maxIndex
    return maxIndex
    
def reshape_fc_weight_matrix(fcWeights, proper_shape):
    total_height, n_filters = fcWeights.shape
    proper_depth, proper_height, proper_width = proper_shape
    temp = np.empty((n_filters, proper_depth, proper_height, proper_width))
    #Each column of an FC weight matrix is a filter that will be placed over the entire input. We want to turn each of them into a proper_height x proper_width matrix, and end up with something of shape (n_filters, proper_depth, proper_height, proper_width). 
    for i in range(n_filters):
        for j in range(proper_depth):
            for k in range(proper_height):
                for l in range(proper_width):
                    index = k*proper_width + l + j
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
    global weightMatrix, symInput
    temp = inputMatrix[inputNumber]
    
    #print inputMatrix.shape[0], "examples,", 
    #print weightMatrix.shape[0], "layers"
    #print "Input shape is", temp.shape
    for i in range(len(weightMatrix)):
        #print "Shape of weight matrix:",weightMatrix[i].shape
        #If we're not doing an FC->Conv conversion, take out this next line
        weightMatrix[i] = reshape_fc_weight_matrix(weightMatrix[i], temp.shape[1:])
        #print "Shape of weight matrix:",weightMatrix[i].shape
        #print "Number of biases (should = number of filters):",len(biasMatrix[i])
        temp = conv_layer_forward(temp, weightMatrix[i], biasMatrix[i], stride, padding)
        #temp = conv_layer_forward_ineff(temp, weightMatrix[i], biasMatrix[i], stride, padding)
        symInput = sym_conv_layer_forward(symInput, weightMatrix[i], biasMatrix[i], stride, padding)
        temp = relu_layer_forward(temp)
        symInput = relu_layer_forward(symInput)
        temp = pool_layer_forward(temp, 1)
        #temp = pool_layer_forward_ineff(temp, 1)
    #print temp
    max_index = classify(temp)
    plt.imshow(symInput[0,0,max_index])
    plt.figure()
    plt.imshow(inputMatrix[inputNumber][0,0,:,:])
    plt.figure()
    plt.imshow(np.multiply(symInput[0,0,max_index], inputMatrix[inputNumber][0,0,:,:]))
    plt.show()
    
'''f=h5py.File('mnist_complicated.h5','r')
model_weights = f['model_weights']
#print model_weights.keys()
layer = 0
conv_layers = [p for p in model_weights.keys() if p.startswith('conv2d')]
dense_layers = [p for p in model_weights.keys() if p.startswith('dense')]
pool_layers = [p for p in model_weights.keys() if p.startswith('max_pool')]
convWeightMatrix = np.empty(len(conv_layers),dtype=list)
convBiasMatrix = np.empty(len(conv_layers),dtype=list)
denseWeightMatrix = np.empty(len(dense_layers),dtype=list)
denseBiasMatrix = np.empty(len(dense_layers),dtype=list)
for k in conv_layers:
    for j in model_weights[k][k].keys():
        if j.startswith('kernel'):
            convWeightMatrix[layer] = np.zeros(model_weights[k][k][j].shape)
            convWeightMatrix[layer] = model_weights[k][k][j]
        if j.startswith('bias'):
            convBiasMatrix[layer] = np.zeros(model_weights[k][k][j].shape)
            convBiasMatrix[layer] = model_weights[k][k][j]
    layer = layer+1
layer = 0
for k in dense_layers:
    for j in model_weights[k][k].keys():
        if j.startswith('kernel'):
            denseWeightMatrix[layer] = np.zeros(model_weights[k][k][j].shape)
            denseWeightMatrix[layer] = model_weights[k][k][j]
        if j.startswith('bias'):
            denseBiasMatrix[layer] = np.zeros(model_weights[k][k][j].shape)
            denseBiasMatrix[layer] = model_weights[k][k][j]
    layer = layer+1'''

def pool_testing(size, stride):
    X = np.arange(16).reshape(4, 4, 1)
    #print X
    h, w, d = X.shape
    h_out = (h-size)/stride + 1
    w_out = (w-size)/stride + 1
    out = np.zeros((h_out, w_out, d))
    for i in range(h_out):
        for j in range(w_out):
            for k in range(d):
                rowIndex = i*stride
                colIndex = j*stride
                print(X[rowIndex:rowIndex+size,colIndex:colIndex+size,k])
                out[i,j,k] = X[rowIndex:rowIndex+size,colIndex:colIndex+size,k].max()
    print(out)
    
    
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
  
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 

def tf_testing_3A():
    graph = tf.Graph()
    with tf.Session() as sess:
        imported_graph = tf.train.import_meta_graph('tf_models1/gradients_testing.meta')
        #imported_graph = tf.train.import_meta_graph('tf_models1/mnist.meta')
        imported_graph.restore(sess, tf.train.latest_checkpoint('./tf_models1'))
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
                
            print(v.shape)
            #temp = sess.run(v)
            #print temp
        for op in graph.get_operations():
            print(op.name)
            print(op.values())
    
def tf_testing_2():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    
    h_conv1 = tf.identity(tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1), name="import/h_conv1")
    #h_conv1 = tf.Print(h_conv1, [h_conv1], message="First convolutional output:\n")
    
    h_pool1 = tf.identity(max_pool_2x2(h_conv1), name="import/h_pool1")
    #h_pool1 = tf.Print(h_pool1, [h_pool1], message="First pooling output:\n")
    
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.identity(tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2), name="import/h_conv2")
    #h_conv2 = tf.Print(h_conv2, [h_conv2], message="Second convolutional output:\n")
    
    h_pool2 = tf.identity(max_pool_2x2(h_conv2),name="import/h_pool2")
    #h_pool2 = tf.Print(h_pool2, [h_pool2], message="Second pooling output:\n")
    
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.identity(tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1), name="import/h_fc1")
    
    keep_prob = tf.placeholder(tf.float32)
    #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    #y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y_conv = tf.identity(tf.matmul(h_fc1, W_fc2) + b_fc2,name="import/y_conv") #No-dropout version
    
    #y_conv = tf.Print(y_conv, [y_conv], message="Final output:\n")
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver({"W_conv1":W_conv1, "b_conv1":b_conv1, "W_conv2":W_conv2, "b_conv2":b_conv2, "W_fc1":W_fc1, "b_fc1":b_fc1, "W_fc2":W_fc2, "b_fc2":b_fc2})#maybe try [W_conv1, b_conv1, h_pool1, W_conv2, b_conv2, h_pool2, W_fc1, b_fc1, h_fc1, W_fc2, b_fc2] as an argument?
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}, session=sess)
                print('step %d, training accuracy %g' % (i, train_accuracy))
                #tf.train.Saver().save(sess, 'tf_models1/mnist_iter', global_step=i)
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}, session=sess)
        
        print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}, session=sess))
    
        saver.save(sess, 'tf_models1/mnist_no_dropout')
        #saver.save(sess, 'tf_models1/mnist')


        imported_graph = tf.train.import_meta_graph('tf_models1/mnist_no_dropout.meta')
        #imported_graph = tf.train.import_meta_graph('tf_models1/mnist.meta')
        imported_graph.restore(sess, tf.train.latest_checkpoint('./tf_models1'))
        graph = tf.get_default_graph()
        
        np.set_printoptions(threshold=np.nan)
        f = open("./example_10.txt", 'r')
        lines = f.readlines()
        thing = str.split(lines[0],',')
        thing = [float(a)+0.5 for a in thing]
        print(str(len(thing)))
        im_data = np.array(thing[1:], dtype=np.float32)
        data = np.ndarray.flatten(im_data)
        feed_dict = {x:[data], keep_prob: 1.0}
      
        tens1 = graph.get_tensor_by_name("import/h_conv1:0")
        print(tens1.name)
        result = tens1.eval(feed_dict)
        print('Conv 1 out:')
        print(result.shape)
        
        tens2 = graph.get_tensor_by_name("import/h_pool1:0")
        print(tens2.name)
        result = tens2.eval(feed_dict)
        print('Pool 1 out:')
        print(result.shape)

        tens3 = graph.get_tensor_by_name("import/h_conv2:0")
        print(tens3.name)
        result = tens3.eval(feed_dict)
        print('Conv 2 out:')
        print(result.shape)

        tens4 = graph.get_tensor_by_name("import/h_pool2:0")
        print(tens4.name)
        result = tens4.eval(feed_dict)
        print('Pool 2 out:')
        print(result.shape)

        tens5 = graph.get_tensor_by_name("import/h_fc1:0")
        print(tens5.name)
        result = tens5.eval(feed_dict)
        print('Dense 1 out:')
        print(result.shape)

        tens6 = graph.get_tensor_by_name("import/y_conv:0")
        print(tens6.name)
        result = tens6.eval(feed_dict)
        print('Final out:')
        print(str(result))
    
def tf_testing_3():
    graph = tf.Graph()
    with tf.Session() as sess:
        imported_graph = tf.train.import_meta_graph('tf_models1/mnist_no_dropout.meta')
        #imported_graph = tf.train.import_meta_graph('tf_models1/mnist.meta')
        imported_graph.restore(sess, tf.train.latest_checkpoint('./tf_models1'))
        graph = tf.get_default_graph()
        tens1 = graph.get_tensor_by_name("import/h_fc1:0")
        print('tens1 h_fc1')
        print(tens1.shape)
        print(tens1.name)
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
            #print(v.name)
            if len(v.shape) == 4: #convolutional layer
                layerTypeList.append('conv2d')
                convWeightMatrix[convLayer] = np.zeros(v.shape)
                convWeightMatrix[convLayer] = sess.run(v)
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
                
            #print(v.shape)
            #temp = sess.run(v)
            #print temp

#temp = inputMatrix[0]

init("./example_10.txt", "./mnist_3A_layer.txt", 28, 28)
#print symInput.shape
#plt.figure()
#plt.imshow(inputMatrix[0][0,0])
#plt.show()
#do_all_layers(9, 1, 0)

#temp = inputMatrix[0]
weightsFile = "./mnist_3A_layer.txt"
inputsFile = "./mnist_test.csv" 
h5File = "./mnist_complicated.h5"
modelFile = "./model.json"
metaFile = "./tf_models1/mnist.meta"
noDropoutMetaFile = "./tf_models1/mnist_no_dropout.meta"
checkpoint = "./tf_models1"
inputIndex = 9


tf_testing_2()
