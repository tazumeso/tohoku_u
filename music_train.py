#-*- coding:utf-8 -*-

import sys
sys.path.append("./deep_learning")

from sys import argv
import glob
import numpy as np
#import matplotlib.pyplot as plt
from deep_learning import *
from deep_learning.conv_net import ConvNet
from deep_learning.optimization import Adam
from deep_learning.dcgan_trainer import DCGAN_trainer

class Generator(ConvNet):
    def __init__(self, nz):
        super(Generator, self).__init__()
        ConvNet.add_affine(self, nz, 256*10*5, output_shape=(256, 10, 5))
        ConvNet.add_batch_normalization(self, 256*10*5, "Relu")
        ConvNet.add_deconv(self, 256, 128, 4, 4, stride=2 ,pad=1, wscale=0.02)
        ConvNet.add_batch_normalization(self, 128*20*10, "Relu")
        ConvNet.add_deconv(self, 128, 64, 4, 4, stride=2 ,pad=1, wscale=0.02)
        ConvNet.add_batch_normalization(self, 64*40*20, "Relu")
        ConvNet.add_deconv(self, 64, 2, 4, 4, stride=2, pad=1, wscale=0.02)
        ConvNet.add_tanh(self)

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
    def __init__(self):
        super(Discriminator,self).__init__()
        ConvNet.add_conv(self, 2, 64, 4, 4, stride=2, pad=1, wscale=0.02)
        ConvNet.add_batch_normalization(self, 64*40*20, "Elu")
        ConvNet.add_conv(self, 64, 128, 4, 4, stride=2, pad=1, wscale=0.02)
        ConvNet.add_batch_normalization(self, 128*20*10, "Elu")
        ConvNet.add_conv(self, 128, 256, 4, 4, stride=2, pad=1, wscale=0.02)
        ConvNet.add_batch_normalization(self, 256*10*5, "Elu")
        ConvNet.add_affine(self, 256*10*5, 100)
        ConvNet.add_batch_normalization(self, 100, "Elu")
        ConvNet.add_affine(self, 100, 2)
        ConvNet.add_softmax(self)

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


def make_image(f, len_mea):
	music_list = np.load(f).transpose(0, 2, 1, 3).tolist()
	music_image_list = []
	for i in range(5, len(music_list) - len_mea + 1):
		music_image = music_list[i]
		for j in range(1, len_mea):
			music_image += music_list[i + j]
		music_image = np.array(music_image)
		if music_image.sum() > 11:
			music_image_list.append(music_image)
			"""
			plt.imshow(music_image.transpose(1, 0, 2)[0])
			plt.show()
			"""
	#print(np.array(music_image_list).shape)
	return music_image_list
	
def make_input(dir_name):
	input_data = []
	for f in glob.glob(dir_name + "/*"):
		input_data += make_image(f, 5)
	return np.array(input_data).transpose(0, 2, 1, 3)
	
#main
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
    
input_data = make_input("xml2vec-master/music_numpy")[:30600]
#100の倍数にしている。
print(input_data.shape)

nz = 100
batch_size = 100
epoch_num = 20  #10000

gen = Generator(nz)
dis = Discriminator()

#gen.save_network("dcgan_faces/Generator_init")
#dis.save_network("dcgan_faces/Discriminator_init")

learning_rate_gen = 2e-4
learning_rate_dis = 1e-5
opt_gen = Adam(lr=0.0002, beta1=0.5)
opt_dis = Adam(lr=0.0001, beta1=0.5)

#学習
dc_trainer = DCGAN_trainer(gen, dis)
train_time = dc_trainer.train(opt_gen, opt_dis, input_data, epoch_num, nz=nz, batch_size=batch_size, save=save, img_test="img_test", graph="graph")


minute = train_time // 60
hour = minute // 60
minute = minute % 60
memo += '\n\n%d時間%d分かかりました' % (hour, minute)

with open(save + "/memo", mode='w') as f:
    f.write(memo)


