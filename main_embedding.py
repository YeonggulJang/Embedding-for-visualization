import matplotlib 
matplotlib.use('Agg')

import os
import tensorflow as tf
import numpy as np
import natsort 		# For natural sorting
import math

import Voxel as vx

from Utility_filenames import Generator, Image_loader, load_imagedata

#import Densenet as densenet
import Densenet_pretrained as densenet

import embedder

# for ROC

from scipy import interp
from sklearn.metrics import roc_curve, auc
from itertools import cycle


def create_training_path(output_path):
	idx = 0
	path = os.path.join(output_path, "run_{:03d}".format(idx))
	while os.path.exists(path):
		idx += 1
		path = os.path.join(output_path, "run_{:03d}".format(idx))
	return path

def find_lastes_training_path(output_path):
	idx = 0
	path = os.path.join(output_path, "run_{:03d}".format(idx))
	while os.path.exists(path):
		idx += 1
		path = os.path.join(output_path, "run_{:03d}".format(idx))
	return os.path.join(output_path, "run_{:03d}".format(idx-1))

def launch(data_path, validation, batchs, epochs, output_path, restore):
	generator = Generator(data_path, batch_size = batchs, val_p = validation)
	
	weights = None

	nX = generator.nX
	nY = generator.nY
	channels = generator.channels
	n_class = generator.n_class

	print nX, nY, channels, n_class
	print generator.nTrain, generator.nValidation
	
	net = densenet.DenseNet(image_size = (nY, nX, channels), n_class = n_class)
	
	path = output_path if restore else create_training_path(output_path)

	trainer = densenet.Trainer(net)
	path = trainer.train(generator, path, batchs = batchs, epochs=epochs, 
						 dropout=0.5, display_step=2, restore=restore)

def preprocessing(data):
	nData = data.shape[0]
	data = data.astype(np.float32)

	mean = np.mean(data, axis=(1, 2, 3))
	stddev = np.std(data, axis=(1, 2, 3))

	for i in range(nData):
		data[i] = (data[i] - mean[i]) / stddev[i]

	return data

def Getnpys():
	# Read the images and concatenate
	images = []
	filenames = []
	#npyDir = '../Testing/Image/ED/'
	npyDir = '../Data/Train/Image/ED/'
	#npyDir = './4D data/resize/'
	for dirName, subdirList, fileList in os.walk(npyDir):
		# Sort the tif file numerically
		fileList = natsort.natsorted(fileList) 

		for f in fileList:
			if f.endswith(".npy"):
				filename, file_extension = os.path.splitext(f)
				fullpath = os.path.join(npyDir,f)

				if any(filename[0:3] in string for string in exclusion_list5):
					print filename
					image = np.load(fullpath)
					images.append(image)
					filenames.append(filename)

	return images, filenames

if __name__ == '__main__':
	import os
	os.environ["CUDA_VISIBLE_DEVICES"]="0"

	
	import matplotlib.pyplot as plt

	data_path = "../data/training"
	batch_size = 20
	validation = 0.1
	epochs = 500

	nX = nY = 256
	channels = 3
	
	output_path = "./DenseNet-train/run_169"
	#restore = True
	restore = False

	save_path = "./CAM"

	# # Training
	# launch(data_path, validation, batch_size, epochs, output_path, restore)

	# Prediction
	# data_path = "../data/test"
	# Images = np.load(data_path + "/test_data.npy")
	# Label = np.load(data_path + "/test_label.npy")

	target_path = "/home/icirc/Desktop/YG/ICT/Data_test_sub.csv"
	folder_path = "/home/icirc/Desktop/YG/ICT/images/"
	files_extension = ['.jpg', '.jpeg', '.png']
	filenames, labels = Image_loader(target_path, folder_path, files_extension) 
	n_class = labels.shape[-1]

	nImage = filenames.shape[0]
	nIter = int(math.ceil(nImage / float(batch_size)))
	print (nImage)

	#model_path = find_lastes_training_path(output_path)
	model_path = "./DenseNet-train/run_169"
	net = densenet.DenseNet(image_size = (nY, nX, channels), n_class = n_class)

	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		net.restore_model(sess, model_path + "/model.ckpt-105")

		nTruePositive = 0
		nTrueNegative = 0
		nTrue = 0
		nFalse = 0
		nCorrect = 0

		nCorrect_per_disease = np.zeros(n_class, dtype=np.float32)
		nTrue_per_disease = np.zeros(n_class, dtype=np.float32)
		nTruePositive_per_disease = np.zeros(n_class, dtype=np.float32)
		nFalse_per_disease = np.zeros(n_class, dtype=np.float32)
		nTrueNegative_per_disease = np.zeros(n_class, dtype=np.float32)

		visualization_disease = 0

		total_dataset = None
		total_labels = None
		total_activations = None
		total_prediction = None
		imagesize_embedding = 32

		for itr in range(nIter):
			start = itr * batch_size 
			end = min(((itr + 1) * batch_size), nImage)

			filename_batch = np.array(filenames[start:end])
			X = load_imagedata(filename_batch, folder_path, nX)
			X = preprocessing(X)

			Y = np.array(labels[start:end])

			y_pred = net.predict(sess, X)
			y_pred_th = y_pred > 0.5
			y_pred_th = np.array(y_pred_th).astype(np.float32)
			
			bat_nCorrect = np.equal(y_pred_th, Y)
			bat_nCorrect = np.array(bat_nCorrect).astype(np.float32)
			bat_accuracy = np.mean(bat_nCorrect, axis=0)
			nCorrect_per_disease += np.sum(bat_nCorrect, axis=0)

			bat_nTrue = np.sum(Y, axis=0)
			nTrue_per_disease += bat_nTrue

			bat_nTruePositive = np.sum(np.multiply(y_pred_th, Y), axis=0)
			nTruePositive_per_disease += bat_nTruePositive

			Not_Y = 1 - Y
			bat_nFalse = np.sum(Not_Y, axis=0)
			nFalse_per_disease += bat_nFalse
			
			Not_y_pred_th = 1 - y_pred_th
			bat_nTrueNegative = np.sum(np.multiply(Not_y_pred_th, Not_Y), axis=0)
			nTrueNegative_per_disease += bat_nTrueNegative

			batch_dataset = load_imagedata(filename_batch, folder_path, imagesize_embedding)
			#batch_labels = Y[:, visualization_disease]
			batch_labels = Y
			activations = sess.run(net.GAP, feed_dict={net.InputImg: X, net.keep_prob: 1., net.is_training: False})
			 
			if total_dataset is None:
				total_dataset = batch_dataset
				total_labels = batch_labels
				total_activations = activations
				total_prediction = y_pred
			else:
				total_dataset = np.append(total_dataset, batch_dataset, axis=0)
				total_labels = np.append(total_labels, batch_labels, axis=0)
				total_activations = np.append(total_activations, activations, axis=0)
				total_prediction = np.append(total_prediction, y_pred, axis=0)
				
		print ("Total accuracy : ")
		print (nCorrect_per_disease / nImage)
		print ("Total sensitivity : ")
		print (nTruePositive_per_disease / nTrue_per_disease)
		print (nTrue_per_disease)
		print ("Total specificity : ")
		print (nTrueNegative_per_disease / nFalse_per_disease)
		print (nFalse_per_disease)

		embedder.summary_embedding(sess=sess, dataset=total_dataset, embedding_list=[total_activations],
						   embedding_path=os.path.join(model_path, 'embedding'), image_size=imagesize_embedding,
						   channel=3, labels=total_labels)

		# Compute ROC curve and ROC area for each class8
		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		for i in range(n_class):
			fpr[i], tpr[i], _ = roc_curve(total_labels[:, i], total_prediction[:, i])
			roc_auc[i] = auc(fpr[i], tpr[i])

		# # Compute micro-average ROC curve and ROC area
		# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
		# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

		# # First aggregate all false positive rates
		# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))

		# # Then interpolate all ROC curves at this points
		# mean_tpr = np.zeros_like(all_fpr)
		# for i in range(n_class):
		# 	mean_tpr += interp(all_fpr, fpr[i], tpr[i])

		# # Finally average it and compute AUC
		# mean_tpr /= n_class

		# fpr["macro"] = all_fpr
		# tpr["macro"] = mean_tpr
		# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

		# # Plot all ROC curves
		fig = plt.figure()
		# plt.plot(fpr["micro"], tpr["micro"],
		# 		label='micro-average ROC curve (area = {0:0.2f})'
		# 			''.format(roc_auc["micro"]),
		# 		color='deeppink', linestyle=':', linewidth=4)

		# plt.plot(fpr["macro"], tpr["macro"],
		# 		label='macro-average ROC curve (area = {0:0.2f})'
		# 			''.format(roc_auc["macro"]),
		# 		color='navy', linestyle=':', linewidth=4)
		lw = 2
		colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'darkgray', 'darkred', 'beige', 'green', 'pink'])
		for i, color in zip(range(n_class), colors):
			plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))

		plt.plot([0, 1], [0, 1], 'k--', lw=lw)
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Some extension of Receiver operating characteristic to multi-class')
		plt.legend(loc="lower right")
		#plt.show()
		plt.savefig('ROC curve.png')
		fig.savefig('ROC curve.png')