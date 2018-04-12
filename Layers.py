import tensorflow as tf
from tensorflow.python.training import moving_averages
import numpy as np

WEIGHT_VARIABLES = 'weight_variables'

# CONV_WEIGHT_DECAY = 0.0001
# FC_WEIGHT_DECAY = 0.0001

CONV_WEIGHT_DECAY = 0
FC_WEIGHT_DECAY = 0

def xavier_init(n_inputs, n_outputs, uniform=True):
	"""Set the parameter initialization using the method described.
	This method is designed to keep the scale of the gradients roughly the same
	in all layers.
	Xavier Glorot and Yoshua Bengio (2010):
			Understanding the difficulty of training deep feedforward neural
			networks. International conference on artificial intelligence and
			statistics.
	Args:
	n_inputs: The number of input nodes into each output.
	n_outputs: The number of output nodes for each input.
	uniform: If true use a uniform distribution, otherwise use a normal.
	Returns:
	An initializer.
	"""
	if uniform:
		# 6 was used in the paper.
		init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
		return tf.random_uniform_initializer(-init_range, init_range)
	else:
		# 3 gives us approximately the same limits as above since this repicks
		# values greater than 2 standard deviations from the mean.
		stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
		return tf.truncated_normal_initializer(stddev=stddev)

def convolution2d(x, filter_size, out_channels, weight_decay = 0.0, strides = 1, stddev=0.02, padding_opt = 'SAME', bias = True, name = "Conv2d"):
	"""
	Convolution 2D
	Args:
		x:              Tensor, 4D BHWC input
		out_chnnels :   integer, channels(or depth) of output tensor
		stddev  :       standard deviation for initializer
		padding :       string, padding option
	Return:
		conv    :       Tensor, Convolution 2D output
	"""

	with tf.variable_scope(name):
		weight_decay = CONV_WEIGHT_DECAY

		if weight_decay > 0:
			regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
		else:
			regularizer = None
		collections = [tf.GraphKeys.GLOBAL_VARIABLES, WEIGHT_VARIABLES]

		w = tf.get_variable('weights', [filter_size, filter_size, x.get_shape()[-1], out_channels],
				regularizer=regularizer,
				collections=collections,
				initializer=tf.truncated_normal_initializer(stddev=stddev))

		conv = tf.nn.conv2d(x, w, strides = [1, strides, strides, 1], padding = padding_opt)

		if bias:
			b = tf.get_variable('biases', [out_channels], initializer=tf.constant_initializer(0.0))
			conv = tf.nn.bias_add(conv, b)

		return conv
def batch_renormalization(x, phase_train, name = "brn"):
	return 1

def batch_normalization(x, phase_train, name = "bn"):
	"""
	Batch normalization on convolutional maps.
	Args:
		x:           Tensor, 4D BHWC or 2D BN input maps 
		phase_train: boolean tf.Varialbe, true indicates training phase
		scope:       string, variable scope
	Return:
		normed:      batch-normalized maps
	"""
	input_shape = x.get_shape()
	input_ndim = len(input_shape)

	with tf.variable_scope(name) as scope:
		beta = tf.get_variable('beta', [input_shape[-1]], initializer=tf.constant_initializer(0.0))
		gamma = tf.get_variable('gamma', [input_shape[-1]], initializer=tf.constant_initializer(1.0))

		axis = list(range(input_ndim - 1))
		
		moving_mean = tf.get_variable('moving_mean', input_shape[-1:], initializer=tf.zeros_initializer(), trainable=False)
		moving_variance = tf.get_variable('moving_variance', input_shape[-1:], initializer=tf.constant_initializer(1.), trainable=False)

		# Define a function to update mean and variance
		def update_mean_var():
			#batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
			batch_mean, batch_var = tf.nn.moments(x, axis, name='moments')
			
			update_moving_mean = moving_averages.assign_moving_average(
				moving_mean, batch_mean, decay=0.99, zero_debias=False)
			update_moving_variance = moving_averages.assign_moving_average(
				moving_variance, batch_var, decay=0.99, zero_debias=False)

			with tf.control_dependencies(
					[update_moving_mean, update_moving_variance]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		# Retrieve variable managing training mode
		mean, var = tf.cond(phase_train, update_mean_var, lambda: (moving_mean, moving_variance))
		normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)

	return normed
		
def max_pool2d(x, n, strides, padding_opt = 'VALID'):
	"""
	max_pooling 2D
	Args:
		x:              Tensor, 4D BHWC input
		n :             integer, pooling size
		padding :       string, padding option
	Return:
		output :           Tensor, max_pooling2D output
	"""
	return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, strides, strides, 1], padding = padding_opt )

def avg_pool_2d(x, kernel_size, stride, padding_opt = 'VALID', name="AvgPool2D"):
	kernel = autoformat_kernel_2d(kernel_size)

	with tf.name_scope(name) as scope:
		output = tf.nn.avg_pool(x, ksize = kernel, strides = [1, stride, stride, 1], padding = padding_opt)

	return output


def global_avg_pool(x, name="GlobalAvgPool"):
	""" 
	Global Average Pooling
	Args:
		x       :  4-D Tensor [batch, height, width, in_channels].
	Output:
		2-D Tensor [batch, in_channels]
	"""
	input_shape = x.get_shape().as_list()
	assert len(input_shape) == 4, "Incoming Tensor shape must be 4-D"

	with tf.name_scope(name):
		output = tf.reduce_mean(x, [1, 2])

	return output

def upsample2d(x, kernel_size = 2):#, method = ResizeMethod.BILINEAR):
	"""
	upsample 2D
	Args:
		x :              Tensor, 4D BHWC input
		kernel_size :    integer, Upsampling kernel size.
		method :         Resize method, default Bilinear interpolation.

						 ResizeMethod.BILINEAR: Bilinear interpolation. 
						 ResizeMethod.NEAREST_NEIGHBOR: Nearest neighbor interpolation. 
						 ResizeMethod.BICUBIC: Bicubic interpolation. 
						 ResizeMethod.AREA: Area interpolation.
	Return:
		output :         Tensor, 4D B new_H new_W C output
	"""

	input_shape = tf.shape(x)
	new_size = input_shape[1:3] * kernel_size

	#output = tf.image.resize_images(x, size = new_size, method = method)
	#output = tf.image.resize_bilinear(x, size = new_size)
	output = tf.image.resize_nearest_neighbor(x, size = new_size)
	
	return output

def pixel_wise_softmax(x):
	"""
	pixel wise softmax
	Args:
		x :             Tensor, 4D BHWC input
	Return:
		output :        Tensor, pixel wise softmax
	"""
	
	# pwMax = tf.reduce_max(x, 3, keep_dims=True)
	# pwMax = tf.tile(pwMax, tf.pack([1, 1, 1, tf.shape(x)[3]]))
	# x = x - pwMax
	
	# exponential_map = tf.exp(x)
	# sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
	# tensor_sum_exp = tf.tile(sum_exp, tf.pack([1, 1, 1, tf.shape(x)[3]]))
	# output = tf.div(exponential_map, tensor_sum_exp)

	pwMax = tf.reduce_max(x, 3, keep_dims=True)
	pwMax = tf.tile(pwMax, tf.stack([1, 1, 1, tf.shape(x)[3]]))
	x = x - pwMax
	
	exponential_map = tf.exp(x)
	sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
	tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(x)[3]]))
	output = tf.div(exponential_map, tensor_sum_exp)
	
	return output

def fc_layer(x, out_size, weight_decay = 0.0, stddev=0.02, bias = True, name=''):
	"""
	fully connected layer
	Args:
		x        :      Tensor, 2D BN input
		out_size :      integer, the number of output nodes
		name     :      string, layer name
	Return:
		output :        Tensor, output nodes
	"""
	with tf.variable_scope(name):
		weight_decay = FC_WEIGHT_DECAY

		if weight_decay > 0:
			regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
		else:
			regularizer = None
		collections = [tf.GraphKeys.GLOBAL_VARIABLES, WEIGHT_VARIABLES]

		w = tf.get_variable('weights', [x.get_shape()[-1], out_size],
				regularizer=regularizer,
				collections=collections,
				initializer=tf.truncated_normal_initializer(stddev=stddev))
		output = tf.matmul(x, w)

		if(bias):
			b = tf.get_variable('biases', [out_size], initializer=tf.constant_initializer(0.0))
			output = output + b
		
		return output

def lrelu(x, leak=0.2, name="lrelu"):
	"""
	fully connected layer
	Args:
		x        :      Tensor, 
		leak     :      float, the gradient of relu
		name     :      string, layer name
	Return:
		output :        Tensor, 
	"""
	return tf.maximum(x, leak * x)

def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d", with_w=False):
	with tf.variable_scope(name):
	# filter : [height, width, output_channels, in_channels]
		w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
			  initializer=tf.random_normal_initializer(stddev=stddev))
		try:
			deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
				strides=[1, d_h, d_w, 1])
		# Support for verisons of TensorFlow before 0.7.0
		except AttributeError:
			deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
				strides=[1, d_h, d_w, 1])

		biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
		deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

		if with_w:
			return deconv, w, biases
		else:
			return deconv

def cross_entropy(pred_y, y):
	"""
	cross_entropy
	Args:
		pred_y        :      Tensor, 2D [Batch, Class] predict y
		y             :      Tensor, 2D [Batch, Class] groundtruth y
	Return:
		output        :      Tensor, scalar xcross_entropy loss
	"""
	return -tf.reduce_sum(y * tf.log(tf.clip_by_value(pred_y, 1e-10, 1.0)), name="xcross_entropy", reduction_indices=1)

def binary_cross_entropy(pred_y, y):
	"""
	cross_entropy
	Args:
		pred_y        :      Tensor, 2D [Batch, 1] predict y
		y             :      float, groundtruth y, 1 or 0
	Return:
		output        :      Tensor, scalar binary cross_entropy loss
	"""
	# For stability, need to combine it with sigmoid function.
	# z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x)) --> max(x, 0) - x * z + log(1 + exp(-abs(x)))

	loss = y * tf.log(tf.clip_by_value(pred_y, 1e-10, 1.0)) \
			+ (1 - y) * tf.log(tf.clip_by_value((1-pred_y), 1e-10, 1.0))
	return -loss

# Auto format kernel
def autoformat_kernel_2d(strides):
	if isinstance(strides, int):
		return [1, strides, strides, 1]
	elif isinstance(strides, (tuple, list, tf.TensorShape)):
		if len(strides) == 2:
			return [1, strides[0], strides[1], 1]
		elif len(strides) == 4:
			return [strides[0], strides[1], strides[2], strides[3]]
		else:
			raise Exception("strides length error: " + str(len(strides))
							+ ", only a length of 2 or 4 is supported.")
	else:
		raise Exception("strides format error: " + str(type(strides)))