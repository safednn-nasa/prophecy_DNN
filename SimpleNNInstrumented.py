#This one is just to implement the small example with manual initiation that Corina and Divya had previously. It doesn't have all the various file-reading, etc. functions that ConvNNInstrumented has, since it's just here as proof-of-concept. Some of the implementations of similarly-named functions are slightly different, since some things are set up differently here. 

import numpy as np;

n0 = 5
n1 = 3
n2 = 2

start = np.array([0.8, 0.4, 0.6, 0.3, 0.2])

layer0 = np.zeros((n0,1,1), dtype=float)
layer1 = np.zeros((n1,1,1), dtype=float)
layer2 = np.zeros((n2,1,1), dtype=float)
sym_layer0 = np.zeros([n0, n0])
sym_layer1 = np.zeros([n1, n0])
sym_layer2 = np.zeros([n2, n0])
weights1 = np.empty([n0, n1])
weights2 = np.empty([n1, n2])

b0 = 0.5
b1 = 0.5

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
    
def im2col_sliding_strided(A, block_shape, stepsize=1):
    m,n,d = A.shape
    print "Im2col input shape:",m, n, d
    s0, s1, s2 = A.strides   
    print "Im2col input strides",s0, s1 
    nrows = m-block_shape[0]+1
    ncols = n-block_shape[1]+1
    print "Im2col output # of rows, cols:",nrows, ncols
    shp = block_shape[0],block_shape[1],nrows,ncols
    print shp
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(block_shape[0]*block_shape[1],-1)[:,::stepsize]

#Let's just assume the image is 2D for now and figure the rest out later. 
def im2col_indices(input, section_height, section_width, padding=1, stride=1):
    input_padded = np.pad(input, ((padding, padding),(padding, padding)), mode='constant')
    k, i, j = get_im2col_indices((1,)+input.shape, section_height, section_width, padding, stride)
    print k, i, j
    cols = input_padded[:, k, i, j]
    print cols
    C = input.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(section_height * section_width * C, -1)
    return cols

def conv_layer_forward(input, filter, b, stride=1, padding=1):
    print "Conv: input shape",input.shape
    h_x, w_x, d_x = input.shape #Still assuming one 2D image at a time for now, so we omit the n_x and d_x should always be 1
    print "Conv: filter shape",filter.shape
    n_filters, h_filter, w_filter = filter.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1
    print "h_out, w_out",h_out,w_out
    input_col = im2col_sliding_strided(input, (h_filter, w_filter), stride) #im2col(input, filter.shape[0], filter.shape[1], padding=padding, stride=stride)
    print "Input_col shape",input_col.shape
    filter_col = filter.reshape(n_filters, -1)
    print "Filter_col shape",filter_col.shape
    converted_biases = np.array(b).reshape(-1, 1)
    
    out = np.add(np.dot(filter_col, input_col), converted_biases)
    out = out.reshape(n_filters, h_out, w_out) #(n_filters, h_out, w_out, n_x) if multi-input
    #out = out.transpose(3, 0, 1, 2) #Turns it back to (n_x, n_filters, h_out, w_out), but we don't need that since we don't have multiple inputs
    print "output shape",out.shape
    return out;
    
#Each node in the layer must have an array with a list input.size of coefficients. Aside from the FC->Convolution conversion case, can we even do that as a matrix operation? For FC layer, it's just a dot product: previous sym_layer with the weight matrix. 
#If our input is X x Y, every point on a given internal/output layer needs an X x Y map of coefficients. Can we create a weight matrix by overlapping the filters by their stride? That would be the total impact a pixel had overall on the next layer, but not appropriately divided.
def sym_conv_layer_forward(input, filter, b, stride=1, padding=1):
    out = np.dot(input.transpose(), filter)
    out = out.transpose()
    return out;
    
def relu_layer_forward(x):
    relu = lambda x: x * (x > 0).astype(float)
    return relu(x);
    
def reshape_fc_weight_matrix(fcWeights, proper_shape):
    total_height, n_filters = fcWeights.shape
    proper_height, proper_width, depth = proper_shape
    temp = np.empty((n_filters, proper_height, proper_width))
    #Each column of an FC weight matrix is a filter that will be placed over the entire input. We want to turn each of them into a proper_height x proper_width matrix, and end up with something of shape (n_filters, proper_height, proper_width). 
    for i in range(n_filters):
        for j in range(proper_width):
            for k in range(proper_height):
                index = j*proper_width + k
                temp[i][k][j] = fcWeights[index][i]
    return temp
    
def init():
    global layer0, sym_layer0, weights1, weights2
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
            
def compute_layer1():
    global sym_layer1, layer1
    layer1 = conv_layer_forward(layer0, reshape_fc_weight_matrix(weights1, (n0, 1, 1)), b0, 1, 0)
    sym_layer1 = sym_conv_layer_forward(sym_layer0, weights1, b0)
    
    layer1 = relu_layer_forward(layer1)
    sym_layer1 = relu_layer_forward(sym_layer1)
    
    for i in range(sym_layer1.shape[0]):
        print "sym_layer1 ",i
        print sym_layer1[i]
    
def compute_layer2():
    global layer2, sym_layer2
    layer2 = conv_layer_forward(layer1, reshape_fc_weight_matrix(weights2, (n1, 1, 1)), b1, 1, 0)
    sym_layer2 = sym_conv_layer_forward(sym_layer1, weights2, b1)
    
    layer2 = relu_layer_forward(layer2)
    sym_layer2 = relu_layer_forward(sym_layer2)
    
    for i in range(sym_layer2.shape[0]):
        print "sym_layer2 ",i
        print sym_layer2[i]
    
def classify():
    maxValue=0
    maxIndex=-1
    for i in range(n2):
        print "Class ",i," confidence: ",layer2[i]
        if(layer2[i] > maxValue):
            maxValue = layer2[i]
            maxIndex = i
        
    print "MaxIndex: ",maxIndex
    print sym_layer2[maxIndex]

init()
print "layer0",layer0
compute_layer1()
print "layer1",layer1
compute_layer2()
print "layer2",layer2
classify()