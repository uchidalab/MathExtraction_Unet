#
#	MathExtraction_Unet.py -- メインファイル
#

import os
import numpy as np
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger
import sys
import cv2
from skimage.morphology import dilation, disk

from unet_param import UNet

# python3 MathExtraction_Unet.py [input image] [output image] [win_size] [layer_num] [input_weight_file]

argv = sys.argv

scale = 0.25
IMAGE_SIZE = int(argv[3])
LAYER_NUM = int(argv[4])

FIRST_LAYER_FILTER_COUNT = 64
image_stride = int ( IMAGE_SIZE/2 )
weight_file = argv[5]

# 値を-1から1に正規化する関数
def normalize_x(image):
	image = image/127.5 - 1
	return image


# 値を0から1に正規化する関数
def normalize_y(image):
	image = image/255
	return image


# 値を0から255に戻す関数
def denormalize_y(image):
	image = image*255
	return image


def denormalize_y_max(image):
	max_pixel = image.max()
	image = image/max_pixel * 255
	return image


# インプット画像を読み込む関数
def load_X(image_path):

	image = cv2.imread ( image_path, cv2.IMREAD_GRAYSCALE )
	h, w = image.shape
	
	image = cv2.bitwise_not ( image )
	image = dilation ( image, selem=disk(int(1 / (scale*2))) )
	
	nh = int ( h * scale )
	nw = int ( w * scale )

	image = cv2.resize( image, (nw,nh) , interpolation = cv2.INTER_CUBIC)
#	image = cv2.copyMakeBorder ( image, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, cv2.BORDER_WRAP)

	nh += IMAGE_SIZE * 2;
	nw += IMAGE_SIZE * 2;

	num_row = nh // image_stride
	num_col = nw // image_stride

	padh = num_row * image_stride + IMAGE_SIZE
	padw = num_col * image_stride + IMAGE_SIZE
	image = cv2.copyMakeBorder ( image, IMAGE_SIZE, (padh-nh)+IMAGE_SIZE, IMAGE_SIZE, (padw-nw)+IMAGE_SIZE, cv2.BORDER_WRAP)
		
	images = np.zeros((num_row * num_col, IMAGE_SIZE, IMAGE_SIZE, 1), np.float32)
	#cv2.imwrite ( 'original.png', image )
	axes = []
	i = 0;
	for row in range ( num_row ):
		for col in range ( num_col ) :
			cropimg = np.copy ( image[row*image_stride:row*image_stride+IMAGE_SIZE,
									  col*image_stride:col*image_stride+IMAGE_SIZE,
									  np.newaxis] )
			images[i] = normalize_y ( cropimg )
			i+=1
			axes.append ( (row*image_stride, col*image_stride ) )

	return images, axes, (padh, padw), (nh, nw), normalize_y ( image[:,:,np.newaxis] )


# 学習後のU-Netによる予測を行う関数
def predict( Input, Output ):

	# Inputで指定された画像を読み込み sliding Window に分ける
	X_test, axes, img_shape, org_shape, org_img = load_X(Input)

	input_channel_count = 1
	output_channel_count = 1
	first_layer_filter_count = FIRST_LAYER_FILTER_COUNT
	network = UNet(input_channel_count, output_channel_count, first_layer_filter_count, IMAGE_SIZE, LAYER_NUM )
	model = network.get_model()
	model.load_weights(weight_file)

	BATCH_SIZE = 8
	Y_pred = model.predict(X_test, BATCH_SIZE, verbose=True)

	result_img = np.zeros ( (img_shape[0], img_shape[1], 1), np.float32 )

	for i, y in enumerate(Y_pred):
		z = result_img[axes[i][0]:axes[i][0]+IMAGE_SIZE, axes[i][1]:axes[i][1]+IMAGE_SIZE, :]
		result_img[axes[i][0]:axes[i][0]+IMAGE_SIZE, axes[i][1]:axes[i][1]+IMAGE_SIZE] = np.maximum ( z, y )

	result = result_img * org_img

	return result[IMAGE_SIZE:org_shape[0]-IMAGE_SIZE,IMAGE_SIZE:org_shape[1]-IMAGE_SIZE]


def postprocess ( pred_img, image_path ):
	image = cv2.imread ( image_path, cv2.IMREAD_GRAYSCALE )
	image = cv2.bitwise_not ( image )

	h, w = image.shape
	pred_resized=cv2.resize(pred_img,(w,h),interpolation=cv2.INTER_CUBIC)
	first_masked = pred_resized * image

	labelnum, labelimg, contours, GoCs = cv2.connectedComponentsWithStats(image)	

	for i in range(labelnum):
		left  = contours[i][cv2.CC_STAT_LEFT]
		right = left + contours[i][cv2.CC_STAT_WIDTH] 
		top   = contours[i][cv2.CC_STAT_TOP]
		bottom = top + contours[i][cv2.CC_STAT_HEIGHT] 
		crop_first = first_masked[top:bottom,left:right]
		crop_label = labelimg[top:bottom,left:right]
		crop_img = image[top:bottom,left:right]

		second_masked = (crop_label == i) * crop_first
		second_area = np.sum ( second_masked/255 )

		if ( second_area / contours[i][cv2.CC_STAT_AREA] ) < 0.7: 
			crop_img[ crop_label == i ] = 0

	tmp_image = dilation ( image, selem=disk(int(1 / (scale*2))) )
	nh = int ( h * scale )
	nw = int ( w * scale )
	tmp_image = cv2.resize( tmp_image, (nw,nh) , interpolation = cv2.INTER_CUBIC)
	result_image = dilation ( tmp_image, selem=disk(1) )

	return (result_image)/255

if __name__ == '__main__':
	
	args = sys.argv

	InputFile = args[1]
	OutputFile = args[2]

	# config = tf.ConfigProto(allow_soft_placement=True)
	# config.gpu_options.allow_growth = True
	# session = tf.Session(config=config)
	# K.set_session(session)	

	
	pred_img = predict( InputFile, OutputFile )

	result_image = postprocess ( pred_img, InputFile )

	cv2.imwrite( OutputFile,  denormalize_y_max(result_image) )
