import numpy as np
from conv_net import ConvNet
from optimization import *
import sys

class Trainer:
	def __init__(self, network):
		self.network = network
		
	def train(self, optimizer, input_data, epoch_num, batch_size=100, save=None):
		x_train = input_data["x_train"]
		t_train = input_data["t_train"]
		x_test = input_data["x_test"]
		t_test = input_data["t_test"]
		train_loss_list = []
		test_acc_list = []
		train_num = len(x_train)
		iter_num_per_epoch = train_num // batch_size  
		for epoch in range(epoch_num):
			mask_list = np.arange(train_num)
			np.random.shuffle(mask_list)
			mask_list = mask_list.reshape(iter_num_per_epoch, batch_size)
			for batch_mask in mask_list:
				x_batch = x_train[batch_mask]
				t_batch = t_train[batch_mask]
						
				grads = self.network.gradient(x_batch, t_batch)
				paras = self.network.paras
				optimizer.update(paras, grads)
				loss = self.network.loss(x_batch, t_batch)
				sys.stdout.write("\r%f" % loss)
				sys.stdout.flush()
				train_loss_list.append(loss)
		
			batch_mask = np.random.choice(len(x_test), 1000)
			x_batch = x_test[batch_mask]
			t_batch = t_test[batch_mask]
			test_acc = self.network.accuracy(x_batch, t_batch)
			test_acc_list.append(test_acc)
			print()   
			print('epoch%i loss : %f' %(epoch + 1, loss))
			print('epoch%i accuracy : %f' %(epoch + 1, test_acc)) 
			if save is not None:
				self.network.save_network(save + "/network_%i" % (epoch+1))
			
		
		return train_loss_list, test_acc_list
		
		