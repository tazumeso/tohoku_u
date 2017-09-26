#coding:utf-8

import sys, os
import time
import numpy as np
from mnist_load import load_mnist
from conv_net import ConvNet
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from optimization import *
import pickle
from trainer import Trainer
start = time.time()

class SimpleConv(ConvNet):
    def __init__(self):
        super(SimpleConv, self).__init__()
        ConvNet.add_conv(self, 1, 30, 5, 5)
        ConvNet.add_batch_normalization(self, 30*24*24, "Relu")
        ConvNet.add_pooling(self, 2, 2, stride=2)
        ConvNet.add_affine(self, 30*12*12, 200)
        ConvNet.add_batch_normalization(self, 200, "Relu")
        ConvNet.add_affine(self, 200, 10)
        ConvNet.add_softmax(self)

(x_train, t_train), (x_test, t_test) = load_mnist(load_file="mnist.pkl", flatten=False, one_hot_label=False)

network = SimpleConv()

optimizer = Adam()

input_data = {"x_train": x_train, "t_train": t_train, "x_test": x_test, "t_test": t_test}

trainer = Trainer(network)
train_loss_list, test_acc_list = trainer.train(optimizer, input_data, 3, batch_size = 100)

elapsed_time = time.time() - start
print("elapsed_time : %f [sec]" % elapsed_time)
plt.plot(train_loss_list)
plt.savefig("result/loss_list_multi_trainer")

plt.figure()
plt.plot(test_acc_list)
plt.savefig("result/test_acc_list_multi_trainer")

with open('network/mnist_network_trainer.pkl', mode='wb') as f:
    pickle.dump(network, f)
