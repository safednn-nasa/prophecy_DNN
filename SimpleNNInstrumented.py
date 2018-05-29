import numpy as np;

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

b0 = 0.5
b1 = 0.5

def im2col_indices(input_shape, field_height, field_width, padding=1, stride=1):
    #N, C, H, W = input_shape
    #print "N, C, H, W: "+N+" "+C+" "+H+" "+W
    C, H, W = input_shape
    print "C, H, W: ",C,H,W
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1
    
    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    
    return (k, i, j)

def im2col(input, section_height, section_width, padding=1, stride=1):
    input_padded = np.pad(input, (padding, padding), mode='constant')
    k, i, j = im2col_indices((1,)+input.shape, section_height, section_width, padding, stride)
    print k, i, j
    cols = input_padded[:, k, i, j]
    print cols
    C = input.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(section_height * section_width * C, -1)
    return cols

def conv_layer_forward(input, filter, b, stride=1, padding=1):
    input_col = input#im2col(input, 1, filter.shape[0], padding=padding, stride=stride)
    filter_col = filter.reshape(1, -1)
    out = np.dot(filter_col, input_col) + b
    #out = out.reshape(filter.shape[0], input.shape[1]) #(n_filters, h_out, w_out, n_x)
    #out = out.transpose(3, 0, 1, 2)
    return out;
    
#Each node in the layer must have an array with a list input.size of coefficients. Aside from the FC->Convolution conversion case, can we even do that as a matrix operation? For FC layer, it's just a dot product: previous sym_layer with the weight matrix. 
#If our input is X x Y, every point on a given internal/output layer needs an X x Y map of coefficients. Can we create a weight matrix by overlapping the filters by their stride? That would be the total impact a pixel had overall on the next layer, but not appropriately divided.
def sym_conv_layer_forward(input, filter, b, stride=1, padding=1):
    out = np.dot(input, filter)
    out = out.transpose()
    return out;
    
def relu_layer_forward(x):
    relu = lambda x: x * (x > 0).astype(float)
    return relu(x);

def pool_layer_forward(X, n, d, h, w, stride = 1):
    X_reshaped = X.reshape(n * d, 1, h, w)
    X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride)
    max_idx = np.argmax(X_col, axis=0)
    out = X_col[max_idx, range(max_idx.size)]
    out = out.reshape(h_out, w_out, n, d)
    out = out.transpose(2, 3, 0, 1)
    return out;
    
def init(input):
    for i in range(n0):
        layer0[i] = input[i]
        sym_layer0[i, i] = 1
        print "sym_layer0 ",i
        print sym_layer0[i]
    
    for i in range(n0):
        for j in range (0, n1):
            weights1[i, j] = 0.5+i+j
        
    for i in range(n1):
        for j in range(0, n2):
            weights2[i, j] = 0.3+i
            
#Question for Divya: If I wanted to do the FC->Conv layer transform, do I just transpose the weight matrix and do a dot prodcut with the current input, plus the bias? I think that might just be it, but why didn't they implement it that way in the java version?
def compute_layer1():
    global sym_layer1, layer1
    layer1 = conv_layer_forward(weights1, layer0, b0)
    sym_layer1 = sym_conv_layer_forward(sym_layer0, weights1, b0)
    
    layer1 = relu_layer_forward(layer1)
    sym_layer1 = relu_layer_forward(sym_layer1)
    
    for i in range(sym_layer1.shape[0]):
        print "sym_layer1 ",i
        print sym_layer1[i]
    
def compute_layer2():
    global layer2, sym_layer2
    layer2 = conv_layer_forward(weights2, layer1, b1)
    sym_layer2 = sym_conv_layer_forward(sym_layer1.transpose(), weights2, b1)
    
    layer2 = relu_layer_forward(layer2)
    sym_layer2 = relu_layer_forward(sym_layer2)
    
    for i in range(sym_layer2.shape[0]):
        print "sym_layer2 ",i
        print sym_layer2[i]
    
def classify():
    maxValue=0
    maxIndex=-1
    for i in range(n2):
        print "Class ",i," confidence: ",layer2[0][i]
        if(layer2[0][i] > maxValue):
            maxValue = layer2[0][i]
            maxIndex = i
        
    print "MaxIndex: ",maxIndex
    print sym_layer2[maxIndex]
    
init(start)
print "layer0",layer0
compute_layer1()
print "layer1",layer1
compute_layer2()
print "layer2",layer2
classify()