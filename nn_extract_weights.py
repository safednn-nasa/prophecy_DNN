#!/usr/bin/python

#Turns caffemodels into something formatted like mnist_3A_layer.txt. Look for models that have linear functions. 

import numpy

from pylab import *

caffe_root = '../'  # this file should be run from {caffe_root}/examples

import os
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

from caffe.proto import caffe_pb2

net = caffe_pb2.NetParameter()

caffemodel = "mnist/benchmarks/mnist_10_layer.caffemodel"
output_file = "./mnist/benchmarks/mnist_10_layer.nn"

with open(caffemodel, 'rb') as f:
    net.ParseFromString(f.read())

output = open(output_file, 'w')

ipLayers = [ 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21 ]

output.write( str( size( ipLayers ) ) + '\n\n' )

for i in ipLayers:
    layer = net.layer[i]
    input_size = layer.blobs[0].shape.dim[1]
    output_size = layer.blobs[0].shape.dim[0]

    output.write( str( input_size ) )
    output.write( ',' )
    output.write( str( output_size ) )
    output.write( '\n' )

    weights = layer.blobs[0].data
    weight_string = ''
    for weight in weights:
        weight_string += str( weight ) + ','
    output.write( weight_string[:-1] + '\n' )

    bias = layer.blobs[1].data
    bias_string = ''
    for one_bias in bias:
        bias_string += str( one_bias ) + ','
    output.write( bias_string[:-1] + '\n' )

    output.write( '\n' )
