#coding:utf-8

import sys, os
import numpy as np
from conv_net import ConvNet
from layer_class import *
from optimization import *
from sys import argv
from dcgan_trainer import DCGAN_trainer
from faces.load_face import load_image
from under_treatment import *

class Generator(ConvNet):
    def __init__(self, dir_name):
        super(Generator, self).__init__()
    
        with open(dir_name + "/layers.txt") as f:
            layer_list = f.read().split("\n")[:-1]
        
        for layer in layer_list:
            layer_split = layer.split("\t")
            self.layer_list.append(layer_split[0])
        
            if layer_split[0].split("_")[0] == "Affine":
                layer_num = layer_split[0].split("_")[1]
                W = np.load("%s/%s_W.npy" % (dir_name, layer_split[0]))
                b = np.load("%s/%s_b.npy" % (dir_name, layer_split[0]))
                self.paras["W" + layer_num] = W
                self.paras["b" + layer_num] = b
                self.layers["layer" + layer_num] =\
                    Affine(W, b, output_shape = eval(layer_split[1].split("=")[1]))
                self.layer_num += 1
                
            if layer_split[0].split("_")[0] == "Conv":
                layer_num = layer_split[0].split("_")[1]
                W = np.load("%s/%s_W.npy" % (dir_name, layer_split[0]))
                b = np.load("%s/%s_b.npy" % (dir_name, layer_split[0]))
                self.paras["W" + layer_num] = W
                self.paras["b" + layer_num] = b
                self.layers["layer" + layer_num] =\
                    Convolution(W, b, stride = int(layer_split[1].split("=")[1]), pad=int(layer_split[2].split("=")[1]))
                self.layer_num += 1

            if layer_split[0].split("_")[0] == "Deconv":
                layer_num = layer_split[0].split("_")[1]
                W = np.load("%s/%s_W.npy" % (dir_name, layer_split[0]))
                b = np.load("%s/%s_b.npy" % (dir_name, layer_split[0]))
                self.paras["W" + layer_num] = W
                self.paras["b" + layer_num] = b     
                self.layers["layer" + layer_num] =\
                    Deconvolution(W, b, stride = int(layer_split[1].split("=")[1]), pad=int(layer_split[2].split("=")[1]))
                self.layer_num += 1


            if layer_split[0].split("_")[0] == "Pooling":
                layer_num = layer_split[0].split("_")[1]
                self.layers["Pooling" + layer_num] =\
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
                self.paras["gamma" + batch_norm_num] = gamma
                self.paras["beta" + layer_num] = beta
                self.layers["BatchNorm" + batch_norm_num] =\
                    BatchNormalization(gamma, beta, init_mu = init_mu, init_std = init_std)
                self.batch_norm_num += 1

            if layer_split[0].split("_")[0] == "Relu":
                layer_num = layer_split[0].split("_")[1]
                self.layers["Relu" + layer_num] = Relu()

            if layer_split[0].split("_")[0] == "Elu":
                layer_num = layer_split[0].split("_")[1]
                self.layers["Elu" + layer_num] = Elu(alpha = float(layer_split[1].split("=")[1]))

            if layer_split[0].split("_")[0] == "Sigmoid":
                layer_num = layer_split[0].split("_")[1]
                self.layers["Sigmoid" + layer_num] = Sigmoid()

            if layer_split[0] == "Softmax":
                self.lastLayer = SoftmaxWithLoss()

    def gradient_gen(self, dout):
        layers = list(self.layers.values())
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


class Discriminator(ConvNet):
    def __init__(self, dir_name):
        super(Discriminator, self).__init__()
    
        with open(dir_name + "/layers.txt") as f:
            layer_list = f.read().split("\n")[:-1]
        
        for layer in layer_list:
            layer_split = layer.split("\t")
            self.layer_list.append(layer_split[0])
        
            if layer_split[0].split("_")[0] == "Affine":
                layer_num = layer_split[0].split("_")[1]
                W = np.load("%s/%s_W.npy" % (dir_name, layer_split[0]))
                b = np.load("%s/%s_b.npy" % (dir_name, layer_split[0]))
                self.paras["W" + layer_num] = W
                self.paras["b" + layer_num] = b
                self.layers["layer" + layer_num] =\
                    Affine(W, b, output_shape = eval(layer_split[1].split("=")[1]))
                self.layer_num += 1
                
            if layer_split[0].split("_")[0] == "Conv":
                layer_num = layer_split[0].split("_")[1]
                W = np.load("%s/%s_W.npy" % (dir_name, layer_split[0]))
                b = np.load("%s/%s_b.npy" % (dir_name, layer_split[0]))
                self.paras["W" + layer_num] = W
                self.paras["b" + layer_num] = b
                self.layers["layer" + layer_num] =\
                    Convolution(W, b, stride = int(layer_split[1].split("=")[1]), pad=int(layer_split[2].split("=")[1]))
                self.layer_num += 1

            if layer_split[0].split("_")[0] == "Deconv":
                layer_num = layer_split[0].split("_")[1]
                W = np.load("%s/%s_W.npy" % (dir_name, layer_split[0]))
                b = np.load("%s/%s_b.npy" % (dir_name, layer_split[0]))
                self.paras["W" + layer_num] = W
                self.paras["b" + layer_num] = b     
                self.layers["layer" + layer_num] =\
                    Deconvolution(W, b, stride = int(layer_split[1].split("=")[1]), pad=int(layer_split[2].split("=")[1]))
                self.layer_num += 1


            if layer_split[0].split("_")[0] == "Pooling":
                layer_num = layer_split[0].split("_")[1]
                self.layers["Pooling" + layer_num] =\
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
                self.paras["gamma" + batch_norm_num] = gamma
                self.paras["beta" + layer_num] = beta
                self.layers["BatchNorm" + batch_norm_num] =\
                    BatchNormalization(gamma, beta, init_mu = init_mu, init_std = init_std)
                self.batch_norm_num += 1

            if layer_split[0].split("_")[0] == "Relu":
                layer_num = layer_split[0].split("_")[1]
                self.layers["Relu" + layer_num] = Relu()

            if layer_split[0].split("_")[0] == "Elu":
                layer_num = layer_split[0].split("_")[1]
                self.layers["Elu" + layer_num] = Elu(alpha = float(layer_split[1].split("=")[1]))

            if layer_split[0].split("_")[0] == "Sigmoid":
                layer_num = layer_split[0].split("_")[1]
                self.layers["Sigmoid" + layer_num] = Sigmoid()

            if layer_split[0] == "Softmax":
                self.lastLayer = SoftmaxWithLoss()

    def back_going(self, y, t):
        loss = self.lastLayer.forward(y, t)
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

        return loss, dout,grad

def load_faces():
    if os.path.exists("./faces/x_train.npy"):
        x_train = np.load("faces/x_train.npy")
        return x_train
    else:
        dir_list = ["solty_boy", "solty_girl", "soy_boy", "soy_girl", "source_boy", "source_girl"]
        t = []
        x_train = np.empty((0, 3, 96, 96), float)
        for dir_name in dir_list:
            x = load_image("faces/" + dir_name)
            x_train = np.vstack((x_train, x))
        x_train = x_train.astype(np.uint8)
        np.save('faces/x_train.npy', x_train)
        return x_train
        

def boost_input(x):
    x = np.vstack((x, rolling(x)))
    x = np.vstack((x, flipping(x)))
    x = np.vstack((x, cropping(x, move_rate=8)))
    return x


nz = 100
batch_size = 100
epoch_num = 40  #10000

try:
    save = argv[1]
except:
    save = input("saveディレクトリを指定してください。")
    
print("コメントを書き込んでください。quitで終了します。")
memo = ''
input_memo = ''
while True:
    input_memo =  input()
    if input_memo == 'quit':
        break
    memo += input_memo + '\n'      

gen = Generator("dcgan_faces/Generator_20")
dis = Discriminator("dcgan_faces/Discriminator_20")

#gen.save_network("dcgan_faces/Generator_init")
#dis.save_network("dcgan_faces/Discriminator_init")

learning_rate_gen = 2e-4
learning_rate_dis = 1e-5
opt_gen = Adam(lr=learning_rate_gen)
opt_dis = Adam(lr=learning_rate_dis)

input_img = load_faces()
input_img = boost_input(input_img) / 255.0

#学習
dc_trainer = DCGAN_trainer(gen, dis)
train_time = dc_trainer.train(opt_gen, opt_dis, input_img, epoch_num, nz=nz, batch_size=batch_size, save=save, img_test="img_test", graph="graph")


minute = train_time // 60
hour = minute // 60
minute = minute % 60
memo += '\n\n%d時間%d分かかりました' % (hour, minute)

with open(save + "/memo", mode='w') as f:
    f.write(memo)