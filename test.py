import numpy as np;

def relu_layer_forward(x):
    relu = lambda x: x * (x > 0).astype(float)
    return relu(x);

#X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])

X = np.array([1, 3, -5, 6, 7])
a = (1, 2)
b = (0,)+ a
#input_padded = np.pad(X, (1,1), mode='constant')

print X