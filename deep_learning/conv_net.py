# coding:utf-8

import sys, os
sys.path.append(os.pardir)
import numpy as np
from layer_class import *
from collections import OrderedDict
import glob


class ConvNet:

    def __init__(self):
        self.layer_num = 0
        self.batch_norm_num = 0
        self.paras = {}
        self.layers = OrderedDict()
        self.layer_list = []

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        self.loss(x, t)
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        # layers.reverse()
        for layer in reversed(layers):
            dout = layer.backward(dout)

        grad = {}
        for i in range(self.layer_num):
            grad["W" + str(i)] = self.layers["layer" + str(i)].dW
            grad["b" + str(i)] = self.layers["layer" + str(i)].db
        for i in range(self.batch_norm_num):
            grad["gamma" + str(i)] = self.layers["BatchNorm" + str(i)].dgamma
            grad["beta" + str(i)] = self.layers["BatchNorm" + str(i)].dbeta
        
        return grad

    def add_conv(self, input_chanel, output_chanel, filter_width, filter_depth, stride=1, pad=0, wscale=1):
        self.paras["W" + str(self.layer_num)] =wscale * np.sqrt(2 / input_chanel * filter_width * filter_depth)\
                                                * np.random.randn(output_chanel, input_chanel, filter_width, filter_depth)
        self.paras["b" + str(self.layer_num)] = np.zeros(output_chanel)
        self.layers["layer" + str(self.layer_num)] =\
            Convolution(self.paras["W" + str(self.layer_num)], self.paras["b" + str(self.layer_num)], stride, pad)
        self.layer_list.append("Conv_" + str(self.layer_num))
        self.layer_num += 1

    def add_deconv(self, input_chanel, output_chanel, filter_width, filter_depth, stride=1, pad=0, wscale=1):
        self.paras["W" + str(self.layer_num)] =wscale * np.sqrt(2 / input_chanel * filter_width * filter_depth)\
                                                * np.random.randn(output_chanel, input_chanel, filter_width, filter_depth)
        self.paras["b" + str(self.layer_num)] = np.zeros(output_chanel)
        self.layers["layer" + str(self.layer_num)] =\
            Deconvolution(self.paras["W" + str(self.layer_num)], self.paras["b" + str(self.layer_num)], stride, pad)
        self.layer_list.append("Deconv_" + str(self.layer_num))
        self.layer_num += 1

    def add_batch_normalization(self, input_size, function):
        self.paras['gamma' + str(self.batch_norm_num)] = np.ones(input_size)
        self.paras['beta' + str(self.batch_norm_num)] = np.zeros(input_size)
        self.layers['BatchNorm' + str(self.batch_norm_num)] =\
            BatchNormalization(self.paras['gamma' + str(self.batch_norm_num)], self.paras['beta' + str(self.batch_norm_num)])
        self.layer_list.append("BatchNorm_" + str(self.batch_norm_num))
        self.layers[function + str(self.batch_norm_num)] = eval(function)()
        self.layer_list.append(function + "_" + str(self.batch_norm_num))
        self.batch_norm_num += 1

    def add_pooling(self, pool_h, pool_w, stride=1, pad=0):
        self.layers["Pooling" + str(self.layer_num)] = Pooling(pool_h, pool_w, stride, pad)
        self.layer_list.append("Pooling_" + str(self.layer_num))

    def add_affine(self, input_size, output_size, output_shape=None):
        self.paras["W" + str(self.layer_num)] = np.sqrt(2 / input_size)\
                                                                   * np.random.randn(input_size, output_size)
        self.paras["b" + str(self.layer_num)] = np.zeros(output_size)
        self.layers["layer" + str(self.layer_num)] =\
            Affine(self.paras["W" + str(self.layer_num)], self.paras["b" + str(self.layer_num)], output_shape=output_shape)
        self.layer_list.append("Affine_" + str(self.layer_num))
        self.layer_num += 1

    def add_softmax(self):
        self.lastLayer = SoftmaxWithLoss()
        self.layer_list.append("Softmax")

    def add_sigmoid(self):
        self.layers["Sigmoid" + str(self.layer_num)] = Sigmoid()
        self.layer_list.append("Sigmoid_" + str(self.batch_norm_num))
        
    def add_tanh(self):
        self.layers["Tanh" + str(self.layer_num)] = Tanh()
        self.layer_list.append("Tanh_" + str(self.batch_norm_num))
        
    def save_network(self, dir_name):
        try:
            os.mkdir(dir_name)
        except:
            pass
        
        layers_txt = ""
        #for key, val in self.layers.items():
        #    print(key)
        for layer in self.layer_list:
            #print(layer)
            layers_txt += layer
            layer_split = layer.split("_", 1)
            
            if layer_split[0] == "Affine":
                layers_txt += "\toutput_shape=%s" % str(self.layers["layer" + layer_split[1]].output_shape)
                W = self.layers["layer" + layer_split[1]].W
                b = self.layers["layer" + layer_split[1]].b
                np.save("%s/%s_W.npy" % (dir_name, layer), W)
                np.save("%s/%s_b.npy" % (dir_name, layer), b)
                
            if layer_split[0] == "Conv":
                layers_txt += "\tstride=%i" % self.layers["layer" + layer_split[1]].stride
                layers_txt += "\tpad=%i" % self.layers["layer" + layer_split[1]].pad
                W = self.layers["layer" + layer_split[1]].W
                b = self.layers["layer" + layer_split[1]].b
                np.save("%s/%s_W.npy" % (dir_name, layer), W)
                np.save("%s/%s_b.npy" % (dir_name, layer), b)
                
            if layer_split[0] == "Deconv":
                layers_txt += "\tstride=%i" % self.layers["layer" + layer_split[1]].stride
                layers_txt += "\tpad=%i" % self.layers["layer" + layer_split[1]].pad
                W = self.layers["layer" + layer_split[1]].W
                b = self.layers["layer" + layer_split[1]].b
                np.save("%s/%s_W.npy" % (dir_name, layer), W)
                np.save("%s/%s_b.npy" % (dir_name, layer), b)
                
            if layer_split[0] == "Pooling":
                layers_txt += "\tpool_h=%i" % self.layers["Pooling" + layer_split[1]].pool_h
                layers_txt += "\tpool_w=%i" % self.layers["Pooling" + layer_split[1]].pool_w
                layers_txt += "\tstride=%i" % self.layers["Pooling" + layer_split[1]].stride
                layers_txt += "\tpad=%i" % self.layers["Pooling" + layer_split[1]].pad
                
            if layer_split[0] == "BatchNorm":
                init_mu = self.layers["BatchNorm" + layer_split[1]].init_mu
                init_std = self.layers["BatchNorm" + layer_split[1]].init_std
                np.save("%s/%s_initmu.npy" % (dir_name, layer), init_mu)
                np.save("%s/%s_initstd.npy" % (dir_name, layer), init_std)       
                gamma = self.layers["BatchNorm" + layer_split[1]].gamma
                beta = self.layers["BatchNorm" + layer_split[1]].beta
                np.save("%s/%s_gamma.npy" % (dir_name, layer), gamma)
                np.save("%s/%s_beta.npy" % (dir_name, layer), beta)
            
            if layer_split[0] == "Elu":
                layers_txt += "\talpha=%f" % self.layers["Elu" + layer_split[1]].alpha
            
            layers_txt += "\n"
        
        with open(dir_name + "/layers.txt","w") as f:
            f.write(layers_txt)

            
def load_network(dir_name):
    network = ConvNet()
    network.layer_list = []
    
    with open(dir_name + "/layers.txt") as f:
        layer_list = f.read().split("\n")[:-1]
        
    for layer in layer_list:
        layer_split = layer.split("\t")
        network.layer_list.append(layer_split[0])
        
        if layer_split[0].split("_")[0] == "Affine":
            layer_num = layer_split[0].split("_")[1]
            W = np.load("%s/%s_W.npy" % (dir_name, layer_split[0]))
            b = np.load("%s/%s_b.npy" % (dir_name, layer_split[0]))
            network.paras["W" + layer_num] = W
            network.paras["b" + layer_num] = b
            network.layers["layer" + layer_num] =\
                Affine(W, b, output_shape = eval(layer_split[1].split("=")[1]))
                
        if layer_split[0].split("_")[0] == "Conv":
            layer_num = layer_split[0].split("_")[1]
            W = np.load("%s/%s_W.npy" % (dir_name, layer_split[0]))
            b = np.load("%s/%s_b.npy" % (dir_name, layer_split[0]))
            network.paras["W" + layer_num] = W
            network.paras["b" + layer_num] = b
            network.layers["layer" + layer_num] =\
                Convolution(W, b, stride = int(layer_split[1].split("=")[1]), pad=int(layer_split[2].split("=")[1]))
                
        if layer_split[0].split("_")[0] == "Deconv":
            layer_num = layer_split[0].split("_")[1]
            W = np.load("%s/%s_W.npy" % (dir_name, layer_split[0]))
            b = np.load("%s/%s_b.npy" % (dir_name, layer_split[0]))
            network.paras["W" + layer_num] = W
            network.paras["b" + layer_num] = b     
            network.layers["layer" + layer_num] =\
                Deconvolution(W, b, stride = int(layer_split[1].split("=")[1]), pad=int(layer_split[2].split("=")[1]))

                
        if layer_split[0].split("_")[0] == "Pooling":
            layer_num = layer_split[0].split("_")[1]
            network.layers["Pooling" + layer_num] =\
                Pooling(pool_h = int(layer_split[1].split("=")[1]), pool_w = int(layer_split[2].split("=")[1]),\
                stride = int(layer_split[3].split("=")[1]), pad = int(layer_split[4].split("=")[1]))
            
            
        if layer_split[0].split("_")[0] == "BatchNorm":
            batch_norm_num = layer_split[0].split("_")[1]
            gamma = np.load("%s/%s_gamma.npy" % (dir_name, layer_split[0]))
            beta = np.load("%s/%s_beta.npy" % (dir_name, layer_split[0]))
            if os.path.exists("%s/%s_initmu.npy" % (dir_name, layer_split[0])):
                init_mu = np.load("%s/%s_initmu.npy" % (dir_name, layer_split[0]))
                init_std = np.load("%s/%s_initstd.npy" % (dir_name, layer_split[0]))
            else:
                init_mu = 0
                init_std = 1
            network.paras["gamma" + batch_norm_num] = gamma
            network.paras["beta" + layer_num] = beta
            network.layers["BatchNorm" + batch_norm_num] =\
                BatchNormalization(gamma, beta, init_mu = init_mu, init_std = init_std)
                
        if layer_split[0].split("_")[0] == "Relu":
            layer_num = layer_split[0].split("_")[1]
            network.layers["Relu" + layer_num] = Relu()
            
        if layer_split[0].split("_")[0] == "Elu":
            layer_num = layer_split[0].split("_")[1]
            network.layers["Elu" + layer_num] = Elu(alpha = float(layer_split[1].split("=")[1]))
        
        if layer_split[0].split("_")[0] == "Sigmoid":
            layer_num = layer_split[0].split("_")[1]
            network.layers["Sigmoid" + layer_num] = Sigmoid()
            
        if layer_split[0] == "Softmax":
            network.lastLayer = SoftmaxWithLoss()
        
    return network

#load_network("save_test")



