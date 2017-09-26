# coding:utf-8

import numpy as np
from conv_net import ConvNet
from optimization import *
import sys, os
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import time


class DCGAN_trainer:
    def __init__(self, gen, dis):
        self.gen = gen
        self.dis = dis

    def train(self, opt_gen, opt_dis, input_img, epoch_num, nz=100, batch_size=100, save=None, img_test=None,
              graph=None):

        if save is not None:
            save_dir = save
            try:
                os.mkdir(save_dir)
            except:
                pass

        if img_test is not None:
            if save is None:
                img_test_dir = img_test
            else:
                img_test_dir = "%s/%s" % (save, img_test)
            try:
                os.mkdir(img_test_dir)
            except:
                pass

        if graph is not None:
            if save is None:
                graph_dir = graph
            else:
                graph_dir = "%s/%s" % (save, graph)
            try:
                os.mkdir(graph_dir)
            except:
                pass

        z_test = np.random.uniform(-1, 1, (batch_size, nz)).astype(np.float32)
        losslist_gen = []
        losslist_dis = []

        start = time.time()
        if img_test is not None:  # 1epochごとに絵を出力,100枚ぐらい
            x = self.gen.predict(z_test)
            fig = plt.figure(figsize=(10, 10))
            for (j, img) in enumerate(x):
                ax = fig.add_subplot(10, 10, j + 1, xticks=[], yticks=[])
                img = np.r_[img, np.zeros((1, *img[0].shape))]
                ax.imshow(img.transpose(1, 2, 0))
            plt.savefig('%s/gen_start.png' % img_test_dir)

        for epoch in range(epoch_num):
            input_img_epoch = shuffle(input_img).reshape(int(len(input_img) / batch_size), batch_size,
                                                         *input_img.shape[1:])
            print("start %s epoch_%i" % (save, epoch + 1))
            # inputをランダムに反転、音楽の場合どうかな？

            for (i, x_in) in enumerate(input_img_epoch):
                z = np.random.uniform(-1, 1, (batch_size, nz)).astype(np.float32)
                x = self.gen.predict(z)
                y = self.dis.predict(x)

                # genの学習
                t = np.ones(batch_size, dtype=np.int8)
                loss_gen, dout_x, _ = self.dis.back_going(y, t)
                grads_gen = self.gen.gradient_gen(dout_x)
                paras_gen = self.gen.paras
                opt_gen.update(paras_gen, grads_gen)
                # disの学習
                t = np.zeros(batch_size, dtype=np.int8)
                loss_dis, _, grads_dis = self.dis.back_going(y, t)
                paras_dis = self.dis.paras
                opt_dis.update(paras_dis, grads_dis)

                t = np.ones(batch_size, dtype=np.int8)
                loss_dis_in, _, grads_dis_in = self.dis.back_going(self.dis.predict(x_in), t)
                opt_dis.update(paras_dis, grads_dis_in)

                losslist_gen.append(loss_gen)
                losslist_dis.append((loss_dis + loss_dis_in) / 2)
                if i % 10 == 9:
                    print('iter%i' % (i + 1))

                if i % 100 == 99:
                    print((time.time() - start) // 60)
                    if img_test is not None:  # 1epochごとに絵を出力,100枚ぐらい
                        x = self.gen.predict(z_test)
                        fig = plt.figure(figsize=(10, 10))
                        for (j, img) in enumerate(x):
                            ax = fig.add_subplot(10, 10, j + 1, xticks=[], yticks=[])
                            img = np.r_[img, np.zeros((1, *img[0].shape))]
                            ax.imshow(img.transpose(1, 2, 0))
                        plt.savefig('%s/gen_epoch%d_iter%d.png' % (img_test_dir, epoch + 1, i + 1))

                    if graph is not None:
                        graph = plt.figure(figsize=(10, 10))
                        p1 = plt.plot(losslist_dis)
                        p2 = plt.plot(losslist_gen)
                        plt.legend((p1[0], p2[0]), ("Discriminator", "Generator"), loc=2)
                        plt.savefig("%s/loss_graph" % (graph_dir))

            if save is not None:
                self.gen.save_network(save_dir + "/Generator_%i" % (epoch + 1))
                self.dis.save_network(save_dir + "/Discriminator_%i" % (epoch + 1))

        return time.time() - start
