import os
import numpy as np
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from unet import UNet
from sklearn import model_selection
from keras.utils import multi_gpu_model
import sys

# python3 main_multiGPU_param.py [dataDir] [win_size] [layer_num] [outweight_file]

argv = sys.argv

DATA_DIR = argv[1]
IMAGE_SIZE = int(argv[2])
LAYER_NUM = int(argv[3])
OUT_WEIGHT = argv[4]

FIRST_LAYER_FILTER_COUNT=64
gpu_count = 1

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


# インプット画像を読み込む関数
def load_X(folder_path):
	import os, cv2
	from progressbar import ProgressBar

	image_files = os.listdir(folder_path)
	image_files.sort()
	print( "loading image files : num = {}".format(len(image_files)))
	pbar = ProgressBar(0,len(image_files))
					   
	images = np.zeros((len(image_files), IMAGE_SIZE, IMAGE_SIZE, 1), np.float16)
	for i, image_file in enumerate(image_files):
		image = cv2.imread(folder_path + os.sep + image_file)
		image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		image = cv2.resize(image_gray, (IMAGE_SIZE, IMAGE_SIZE))
		image = image[:,:,np.newaxis]
		images[i] = normalize_y(image)
		pbar.update(i)
	pbar.finish()
	return images, image_files
	

# ラベル画像を読み込む関数
def load_Y(folder_path):
	import os, cv2
	from progressbar import ProgressBar
					   
	image_files = os.listdir(folder_path)
	image_files.sort()
	print( "loading groundtruth files : num = {}".format(len(image_files)))
	pbar = ProgressBar(0,len(image_files))
					   
	images = np.zeros((len(image_files), IMAGE_SIZE, IMAGE_SIZE, 1), np.float16)
	for i, image_file in enumerate(image_files):
		image = cv2.imread(folder_path + os.sep + image_file, cv2.IMREAD_GRAYSCALE)
		image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
		image = image[:,:,np.newaxis]
		images[i] = normalize_y(image)
		pbar.update(i)
	pbar.finish()					   
	return images


# ダイス係数を計算する関数
def dice_coef(y_true, y_pred):
	y_true = K.flatten(y_true)
	y_pred = K.flatten(y_pred)
	intersection = K.sum(y_true * y_pred)
	return 2.0 * intersection / (K.sum(y_true) + K.sum(y_pred) + 1)


def dice_coef_eval ( y_true, y_pred ):
	intersection = sum(y_true.flatten() * y_pred.flatten())
	return 2.0 * intersection / (sum(y_true.flatten()) + sum(y_pred.flatten()) + 1)

	
# ロス関数
def dice_coef_loss(y_true, y_pred):
	return 1.0 - dice_coef(y_true, y_pred)


# U-Netのトレーニングを実行する関数
def train_unet():
	# trainingDataフォルダ配下にleft_imagesフォルダを置いている
	X_train, file_names = load_X(DATA_DIR + os.sep + 'images')
	# trainingDataフォルダ配下にleft_groundTruthフォルダを置いている
	Y_train = load_Y(DATA_DIR + os.sep + 'groundtruth')

	# 入力はグレースケール１チャンネル
	input_channel_count = 1
	# 出力はグレースケール1チャンネル
	output_channel_count = 1
	# 一番初めのConvolutionフィルタ枚数は64
	first_layer_filter_count = FIRST_LAYER_FILTER_COUNT
	# U-Netの生成
	print ( "U-net model generation" )
	network = UNet(input_channel_count, output_channel_count, first_layer_filter_count, IMAGE_SIZE, LAYER_NUM)
	model = network.get_model()

	if os.path.exists ( OUT_WEIGHT  ):
		print ("loading pretrained model..." )
		model.load_weights( OUT_WEIGHT )
		print ("Done." )

	multi_model = multi_gpu_model ( model, gpus = gpu_count )
	multi_model.compile(loss=dice_coef_loss, optimizer=Adam(lr=0.01), metrics=[dice_coef])

	print ( "Data splitting..." )

	# train and verification
	( x_train, x_valid, 
	  y_train, y_valid ) = model_selection.train_test_split ( X_train, Y_train, test_size=0.25 )

	del X_train
	del Y_train

	callbacks = []
	# 学習途中のモデルを残す（ディスク容量を食うので注意）
	callbacks.append ( ModelCheckpoint ( filepath="model_on_train/model_{epoch:03d}.hdf5" ) )
	# 学習過程を CSV 出力
	callbacks.append ( CSVLogger ( "train_history.csv" ) )
	callbacks.append ( EarlyStopping() )	
	BATCH_SIZE = 5 * gpu_count
	# 10エポック
	NUM_EPOCH = 10
	history = multi_model.fit(x_train, y_train, 
							  batch_size=BATCH_SIZE, epochs=NUM_EPOCH, verbose=1, 
							  validation_data = (x_valid, y_valid), callbacks=callbacks )
	model.save_weights(OUT_WEIGHT)

if __name__ == '__main__':

	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth = True
	session = tf.Session(config=config)
	K.set_session(session)	
	
	train_unet()
