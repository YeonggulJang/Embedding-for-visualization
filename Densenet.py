from __future__ import absolute_import, division, print_function

import os
import sys
import shutil
import math
import time

import tensorflow as tf
import numpy as np

from collections import OrderedDict
from Utility_filenames import Reprinter

import Layers as layers

import h5import

CONV_WEIGHT_DECAY = 0.0001
FC_WEIGHT_DECAY = 0.0001

class DenseNet():
	def __init__(self, image_size = (256, 256, 3), n_class = 8, nInit_features = 64, growth_rate = 32, theta = 1.0, dropout_prob = 0.5, bn_size = 4, block_config = (6, 12, 24, 16)):
		"""
		n_class				:   int, the number of target classes
		growth_rate         :   int, parameter k as the growth rate of the network
		theta				:	compression factor
		block_config        :   list, configuration for each block
		dropout_prob        :   float, the initial probability for dropout
		"""
		self.nY = nY = image_size[0]
		self.nX = nX = image_size[1]
		self.nchannels = nchannels = image_size[2]
		
		self.n_class = n_class
		self.nInit_features = nInit_features
		self.growth_rate = growth_rate
		self.theta = theta = 0.5
		self.dropout_prob = dropout_prob
		self.block_config = block_config

		self.bn_size = bn_size

		self.InputImg = tf.placeholder(tf.float32, shape=[None, nY, nX, self.nchannels], name="Input_Image")
		self.Target = tf.placeholder(tf.float32, shape=[None, self.n_class], name="Label")

		# The placeholder dropout and batch_norm        
		self.keep_prob = tf.placeholder(tf.float32, name="Pro_drop_out") # the probability for dropout (keep probability)
		self.is_training = tf.placeholder(tf.bool, name="Is_Training") # the train mode

		# Build dense network.
		self.predicter, self.logits = self.build_network(self.InputImg)

		self.build_class_map(self.bGAP, self.nX)

		# Calculate losses only using logits for numerically stable optimization
		with tf.variable_scope('losses') as scope:
			""" Calculate weight for each class in a batch """
			nSamples = tf.shape(self.Target)[0] # batch size

			nPositive = tf.reduce_sum(self.Target, axis = 0, keepdims = True) # batch x nClasses -> nClasses
			nPositive = tf.tile(nPositive, tf.stack([nSamples, 1])) # nClasses -> batch x nClasses

			nNegative = tf.reduce_sum((1-self.Target), axis = 0, keepdims = True) # batch x nClasses -> nClasses
			nNegative = tf.tile(nNegative, tf.stack([nSamples, 1])) # nClasses -> batch x nClasses

			nSamples = tf.cast(nSamples, tf.float32)
			weight = (self.Target * nPositive + (1-self.Target) * nNegative)/nSamples # z * nPos/nSample + (1-z) * nNeg/nSample
			weight = tf.reciprocal(weight) # y=1/x

			""" Output and loss with class weight """
			loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = self.logits, labels = self.Target)
			loss = tf.multiply(loss, weight)
			self.loss = tf.reduce_sum(loss)

			# With regularization term
			regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
			self.total_loss = tf.add_n([self.loss] + regularization_losses)

		# Accuracy
		with tf.variable_scope('accuracy') as scope:
			predicted_class = tf.greater(self.predicter, 0.65)
			predicted_class = tf.cast(predicted_class, tf.float32)
			correct = tf.equal(predicted_class, self.Target)

			self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name = "accuracy")

			nCorrect = tf.reduce_sum(tf.multiply(predicted_class, self.Target))
			nTotal = tf.reduce_sum(self.Target)
			self.accuracy_for_target = nCorrect/nTotal

	def build_network(self, input_tensor):
		x = layers.convolution2d(input_tensor, filter_size = 7, out_channels = 2 * self.growth_rate, strides = 2, 
								weight_decay = CONV_WEIGHT_DECAY, bias = False, name = 'Convolution0')
		x = layers.batch_normalization(x, self.is_training, name = 'batch0')
		x = tf.nn.relu(x)
		x = layers.max_pool2d(x, 3, 2, padding_opt = 'VALID')

		x = self.dense_block(input_tensor = x, nlayers = 6, name = 'Dense_Block_1')
		x = self.transition_layer(x, name = 'Trans_Layer_1')

		x = self.dense_block(input_tensor = x, nlayers = 12, name = 'Dense_Block_2')
		x = self.transition_layer(x, name = 'Trans_Layer_2')

		x = self.dense_block(input_tensor = x, nlayers = 32, name = 'Dense_Block_3')
		x = self.transition_layer(x, name ='Trans_Layer_3')

		x = self.dense_block(input_tensor = x, nlayers = 32, name = 'Dense_Block_4')

		# Final batch_norm and relu
		x = layers.batch_normalization(x, self.is_training, name = 'final_batch')
		self.bGAP = x = tf.nn.relu(x)

		# Classfication layer
		self.GAP = x = layers.global_avg_pool(x)
		logits = layers.fc_layer(x, self.n_class, weight_decay = CONV_WEIGHT_DECAY, name='logits')
		final = tf.nn.sigmoid(logits)

		return final, logits

	def bottleneck_layer(self, x, name):
		with tf.variable_scope(name):
			x = layers.batch_normalization(x, self.is_training, name = 'batch1')
			x = tf.nn.relu(x)
			x = layers.convolution2d(x, 1, self.bn_size * self.growth_rate, weight_decay = CONV_WEIGHT_DECAY, bias = False, name = 'conv1')
			x = tf.nn.dropout(x, self.keep_prob)

			x = layers.batch_normalization(x, self.is_training, name = 'batch2')
			x = tf.nn.relu(x)
			x = layers.convolution2d(x, 3, self.growth_rate, weight_decay = CONV_WEIGHT_DECAY, bias = False, name = 'conv2')
			x = tf.nn.dropout(x, self.keep_prob)
			return x

	def transition_layer(self, x, name):
		with tf.variable_scope(name):
			input_tensor_depth = int(x.get_shape()[-1])
			output_depth = int(self.theta * input_tensor_depth)
			x = layers.batch_normalization(x, self.is_training, name = 'batch')
			x = tf.nn.relu(x)
			x = layers.convolution2d(x, 1, output_depth, weight_decay = CONV_WEIGHT_DECAY, bias = False, name = 'conv')
			x = tf.nn.dropout(x, self.keep_prob)
			x = layers.avg_pool_2d(x, kernel_size = 2, stride = 2, name="AvgPool2D")
			return x

	def dense_block(self, input_tensor, nlayers, name):
		with tf.variable_scope(name):
			layers_concat = list()
			layers_concat.append(input_tensor)

			x = self.bottleneck_layer(input_tensor, name = 'bottleN_' + str(0))
			layers_concat.append(x)

			for i in range(nlayers - 1):
				x = tf.concat(layers_concat, axis = 3)
				x = self.bottleneck_layer(x, name = 'bottleN_' + str(i + 1))
				layers_concat.append(x)

			x = tf.concat(layers_concat, axis = 3)
			return x

	def save_model(self, session, model_path, epoch):
		"""
		Restores a session from a checkpoint

		Args:
			session     : current session instance
			model_path  : path to file system checkpoint location
		Return:
			save_path   : save path
		"""
		saver = tf.train.Saver()
		save_path = saver.save(session, model_path, global_step = epoch)
		return save_path

	def restore_model(self, session, model_path):
		"""
		Restores a session from a checkpoint

		Args:
			sess        : current session instance
			model_path  : path to file system checkpoint location
		"""
		saver = tf.train.Saver()
		saver.restore(session, model_path)
		print("Model restored from file: %s" % model_path)

	def predict(self, session, data):
		"""
		Uses the model to create a prediction for the given data

		Args:
			data                : Data to predict on. Shape [n, nY, nX, nC]
		Return:
			prediction          : prediction result
		"""
		start_time = time.time()
		prediction = session.run(self.predicter, feed_dict={self.InputImg: data, self.keep_prob: 1., self.is_training: False})
		print("--- %s seconds ---" % (time.time() - start_time))
		return prediction

	def get_heatmap(self, session, data):
		"""
		Uses the model to create a prediction and heat map for the given data

		Args:
			data                : Data to predict on. Shape [n, nY, nX, nC]
			Target_label_idx	: Target label index for localization 
		Return:
			CAM					: class activation map
		"""
		
		conv_resize, weight = session.run([self.conv_resize, self.weight], 
					feed_dict={self.InputImg: data, self.keep_prob: 1., self.is_training: False})
		return conv_resize, weight

	def build_class_map(self, conv, im_width_height):
		output_channels = int(conv.get_shape()[-1])
		conv_resized = tf.image.resize_bilinear(conv, [im_width_height, im_width_height])
		self.conv_resize = conv_resized
		
		with tf.variable_scope('logits', reuse=True):
			self.weight = tf.get_variable('weights')

class Trainer(object):
	"""
	Train DCGAN instance

	parameters:
		net         : the unet instance to train
		batch_size  : size of training batch
		optimizer   : (optional) name of the optimizer to use (momentum or adam)
		opt_kwargs  : (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer
	"""
	prediction_path = "prediction"
	
	def __init__(self, net, batch_size = 16, opt_kwargs={}):
		self.net = net
		self.batch_size = batch_size
		self.opt_kwargs = opt_kwargs
		
	def _get_optimizer(self, training_iters, global_step):
		#opt = tf.train.AdamOptimizer(learning_rate=0.00001, name='Opt').minimize(self.net.total_loss)
		opt = tf.train.AdamOptimizer(learning_rate=0.05, name='Opt').minimize(self.net.total_loss)
		return opt
		
	def _initialize(self, training_iters, output_path, restore):
		step = tf.Variable(0, trainable=False)
		
		with tf.name_scope("training"):
			with tf.name_scope("loss"):
				tf.summary.scalar('Loss', self.net.loss)
				tf.summary.scalar('Total_loss', self.net.total_loss)
			with tf.name_scope("acc"):
				tf.summary.scalar('Accuracy', self.net.accuracy)
				tf.summary.scalar('Accuracy_For_Target', self.net.accuracy_for_target)

		self.optimizer = self._get_optimizer(training_iters, step)
		
		self.summary_op = tf.summary.merge_all()        
		
		#init = tf.initialize_all_variables()
		init = tf.global_variables_initializer()
		
		prediction_path = os.path.abspath(self.prediction_path)
		output_path = os.path.abspath(output_path)

		if not restore:
			print("Removing '{:}'".format(prediction_path))
			shutil.rmtree(prediction_path, ignore_errors=True)
			print("Removing '{:}'".format(output_path))
			shutil.rmtree(output_path, ignore_errors=True)
		
		if not os.path.exists(prediction_path):
			print("Allocating '{:}'".format(prediction_path))
			os.makedirs(prediction_path)
		
		if not os.path.exists(output_path):
			print("Allocating '{:}'".format(output_path))
			os.makedirs(output_path)
		
		return init
		
	def train(self, data_provider, output_path, batchs = 15, epochs=100, dropout=0.6, display_step=1, restore=False):
		"""
		Lauches the training process

		Args:
			data_provider       : callable returning training/validation data and data without label
			output_path         : path where to store checkpoints
			batchs              : number of batchs
			epochs              : number of epochs
			dropout             : dropout probability
			display_step        : number of steps till outputting stats
			restore             : Flag if previous model should be restored 
		"""
		save_path = os.path.join(output_path, "model.ckpt")
		if epochs == 0:
			return save_path
			
		training_iters = int(math.ceil(data_provider.nTrain / float(batchs)))
		validation_iters = int(math.ceil(data_provider.nValidation / float(batchs)))
		
		init = self._initialize(training_iters, output_path, restore)

		reprinter = Reprinter()
		
		with tf.Session() as sess:
			sess.run(init)

			if restore:
				self.net.restore_model(sess, output_path + '/model.ckpt-330')
				# ckpt = tf.train.get_checkpoint_state(output_path)
				# if ckpt and ckpt.model_checkpoint_path:
				# 	print (ckpt.model_checkpoint_path)
				# 	self.net.restore_model(sess, ckpt.model_checkpoint_path)
					
					
			self.coord = tf.train.Coordinator()

			summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)
			
			data_provider.start(self.coord)
			
			print("Start optimization")
			
			try:
				for epoch in range(0, epochs):
					total_loss = 0
					train_acc = 0
					
					for itr in range(training_iters):
						start = itr * batchs
						end = min(((itr + 1) * batchs), data_provider.nTrain)
						step = (itr + 1) + (epoch + 1) * training_iters

						batch = data_provider.next()
						
						_, loss, acc, summary_str = sess.run([self.optimizer, self.net.loss, self.net.accuracy, self.summary_op], 
												feed_dict={self.net.InputImg: batch['X'], self.net.Target: batch['Y'],
														self.net.keep_prob: dropout, self.net.is_training: True})				                   

						if (step % 100 == 0):
							summary_writer.add_summary(summary_str, step)
							summary_writer.flush()
						
						total_loss += loss
						train_acc += acc

						if(step % 10 == 0):
							reprinter.reprint(("Loss = {:.4f}, accuracy = {:.4f}, Iter {:}/{:}\n").format(loss, acc, end, data_provider.nTrain))                  
							sys.stdout.flush()

					self.output_epoch_stats(epoch + 1, epochs, total_loss, train_acc, training_iters)
					reprinter.reset_text()

					val_loss = 0
					val_acc = 0
					for itr in range(validation_iters):
						val_batch = data_provider.get_val_data(itr)
						summary_str, loss, acc = sess.run([self.summary_op, self.net.loss, self.net.accuracy], 
																feed_dict={self.net.InputImg: batch['X'], self.net.Target: batch['Y'], 
																self.net.keep_prob: 1., self.net.is_training: False})

						# val_writer.add_summary(summary_str, step)
						# val_writer.flush()
						
						val_loss += loss
						val_acc += acc
					if(validation_iters > 0):
						print ("Epoch: {:3d}/{:3d}, validation avg loss = {:.4f}, avg acc = {:.4}".format(epoch + 1, epochs, (val_loss / validation_iters), (val_acc / validation_iters)))

					if ((epoch % 15 == 0 and epoch >= 15) or (epoch == epochs - 1)):
						save_path = os.path.join(output_path, "model.ckpt")
						save_path = self.net.save_model(sess, save_path, epoch)

				print("Optimization Finished!")
			except Exception as e: 
				print(e)
			finally:
				data_provider.interrupt()
				return save_path
		
	def store_prediction(self, sess, batch_x, batch_y, name):
		prediction, loss, acc = sess.run((self.net.predicter, self.net.Seg_loss, self.net.accuracy), 
															feed_dict={self.net.InputImg_with_label: batch_x, 
															self.net.Target_Label: batch_y, 
															self.net.keep_prob: 1.,
															self.net.is_training: False})

		#print("Verification erro r= {:.1f}%, loss= {:.4f}".format(error_rate(prediction,batch_y), loss))
				
		# img = util.combine_img_prediction(batch_x, batch_y, prediction)
		# util.save_image(img, "%s/%s.jpg"%(self.prediction_path, name))

		# return pred_shape

	def output_epoch_stats(self, epoch, epochs, total_loss, accuracy, training_iters):
		print ("Epoch : %3d/%3d, Loss = %.4f, accuracy = %.4f" % (epoch, epochs, (total_loss / training_iters), (accuracy / training_iters)), end="\n")
		#print ("Epoch : {:3d}/{:3d}, Loss = {:.4f}, accuracy = {:.4f}".format(epoch, epochs, (total_loss / training_iters), (accuracy / training_iters)))
		sys.stdout.flush()