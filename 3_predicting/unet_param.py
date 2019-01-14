#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers import LeakyReLU, BatchNormalization, Activation, Dropout

class UNet(object):

	def __init__(self, input_channel_count, output_channel_count, first_layer_filter_count, image_size, layer_num ):
		self.INPUT_IMAGE_SIZE = image_size
		self.CONCATENATE_AXIS = -1
		self.CONV_FILTER_SIZE = 4
		self.CONV_STRIDE = 2
		self.CONV_PADDING = (1, 1)
		self.DECONV_FILTER_SIZE = 2
		self.DECONV_STRIDE = 2

		# input (image_size x image_size x input_channel_count)
		inputs = Input((self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, input_channel_count))

		# エンコーダーの作成
		# Encoder 1 (is/2 x is/2 x N)
		enc1 = ZeroPadding2D(self.CONV_PADDING)(inputs)
		enc1 = Conv2D(first_layer_filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(enc1)
		
		# Encoder 2 (is/4 x is/4 x 2N)
		filter_count = first_layer_filter_count*2
		enc2 = self._add_encoding_layer(filter_count, enc1)
		
		# Encoder 3 (is/8 x is/8 x 2N)
		enc3 = self._add_encoding_layer(filter_count, enc2)
		
		# Encoder 4 (is/16 x is/16 x 4N)
		filter_count = first_layer_filter_count*4
		enc4 = self._add_encoding_layer(filter_count, enc3)
		
		# Encoder 5 (is/32 x is/32 x 4N)
		enc5 = self._add_encoding_layer(filter_count, enc4)		
		
		if layer_num >= 6 :
			# Encoder 6 (is/64 x is/64 x 8N)
			filter_count = first_layer_filter_count*8
			enc6 = self._add_encoding_layer(filter_count, enc5)
			
			if layer_num >= 7 :
				# Encoder 7 (is/128 x is/128 x 8N)
				enc7 = self._add_encoding_layer(filter_count, enc6)
				
				if layer_num >= 8 :
					# Encoder 8 (is/256 x is/256 x 16N)
					filter_count = first_layer_filter_count*16
					enc8 = self._add_encoding_layer(filter_count, enc7)
					
					if layer_num >= 9 :
						# Encoder 9 (is/512 x is/523 x 16N)
						enc9 = self._add_encoding_layer(filter_count, enc8)
						
						if layer_num >= 10 :
							# Encoder 10 (is/1024 x is/1024 x 32N)
							filter_count = first_layer_filter_count*32	
							enc10 = self._add_encoding_layer(filter_count, enc9)
							
							# デコーダーの作成
							# Decoder 9 (is/512 x is/523 x 16N)
							filter_count = first_layer_filter_count*16
							dec9 = self._add_decoding_layer(filter_count, True, enc10)
							dec9 = concatenate([dec9, enc9], axis=self.CONCATENATE_AXIS)
						else :
							dec9 = enc9
						
						# Decoder 8 (is/512 x is/523 x 16N)
						dec8 = self._add_decoding_layer(filter_count, True, dec9)
						dec8 = concatenate([dec8, enc8], axis=self.CONCATENATE_AXIS)
					
					else:
						dec8 = enc8
					
					# Decoder 7 (is/128 x is/128 x 8N)
					filter_count = first_layer_filter_count*8
					dec7 = self._add_decoding_layer(filter_count, True, dec8)
					dec7 = concatenate([dec7, enc7], axis=self.CONCATENATE_AXIS)
				else :
					dec7 = enc7

				# Decoder 6 (is/64 x is/64 x 8N)
				dec6 = self._add_decoding_layer(filter_count, False, dec7)
				dec6 = concatenate([dec6, enc6], axis=self.CONCATENATE_AXIS)
			else:
				dec6 = enc6

			# Decoder 5 (is/32 x is/32 x 4N)
			filter_count = first_layer_filter_count*4
			dec5 = self._add_decoding_layer(filter_count, False, dec6)
			dec5 = concatenate([dec5, enc5], axis=self.CONCATENATE_AXIS)

		else:
			dec5 = enc5
		
		# Decoder 4 (is/16 x is/16 x 4N)
		dec4 = self._add_decoding_layer(filter_count, False, dec5)
		dec4 = concatenate([dec4, enc4], axis=self.CONCATENATE_AXIS)

		# Decoder 3 (is/8 x is/8 x 2N)
		filter_count = first_layer_filter_count*2
		dec3 = self._add_decoding_layer(filter_count, False, dec4)
		dec3 = concatenate([dec3, enc3], axis=self.CONCATENATE_AXIS)
		
		# Decorder 2 (is/4 x is/4 x 2N)
		dec2 = self._add_decoding_layer(filter_count, False, dec3)
		dec2 = concatenate([dec2, enc2], axis=self.CONCATENATE_AXIS)
		
		# Decoder 1 (is/2 x is/2 x N )
		filter_count = first_layer_filter_count
		dec1 = self._add_decoding_layer(filter_count, False, dec2)
		dec1 = concatenate([dec1, enc1], axis=self.CONCATENATE_AXIS)
		
		# output layer ( image_size x image_size x output_channel_count)
		outputs = Activation(activation='relu')(dec1)
		outputs = Conv2DTranspose(output_channel_count, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE)(outputs)
		outputs = Activation(activation='sigmoid')(outputs)
		
		self.UNET = Model(inputs=inputs, outputs=outputs)



	# # # Unet 128x128 : 7 layers  ==============================
	# def __init__(self, input_channel_count, output_channel_count, first_layer_filter_count):
	# 	self.INPUT_IMAGE_SIZE = 512
	# 	self.CONCATENATE_AXIS = -1
	# 	self.CONV_FILTER_SIZE = 4
	# 	self.CONV_STRIDE = 2
	# 	self.CONV_PADDING = (1, 1)
	# 	self.DECONV_FILTER_SIZE = 2
	# 	self.DECONV_STRIDE = 2

	# 	# (128 x 128 x input_channel_count)
	# 	inputs = Input((self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, input_channel_count))

	# 	# エンコーダーの作成
	# 	# (64 x 64 x N)
	# 	enc1 = ZeroPadding2D(self.CONV_PADDING)(inputs)
	# 	enc1 = Conv2D(first_layer_filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(enc1)

	# 	# (32 x 32 x 2N)
	# 	filter_count = first_layer_filter_count*2
	# 	enc2 = self._add_encoding_layer(filter_count, enc1)

	# 	# (16 x 16 x 4N)
	# 	filter_count = first_layer_filter_count*4
	# 	enc3 = self._add_encoding_layer(filter_count, enc2)

	# 	# (8 x 8 x 8N)
	# 	filter_count = first_layer_filter_count*8
	# 	enc4 = self._add_encoding_layer(filter_count, enc3)

	# 	# (4 x 4 x 8N)
	# 	enc5 = self._add_encoding_layer(filter_count, enc4)

	# 	# (2 x 2 x 8N)
	# 	enc6 = self._add_encoding_layer(filter_count, enc5)

	# 	# (1 x 1 x 8N)
	# 	enc7 = self._add_encoding_layer(filter_count, enc6)

	# 	# # (1 x 1 x 8N)
	# 	# enc8 = self._add_encoding_layer(filter_count, enc7)

	# 	# デコーダーの作成
	# 	# (2 x 2 x 8N)
	# 	dec1 = self._add_decoding_layer(filter_count, True, enc7)
	# 	dec1 = concatenate([dec1, enc6], axis=self.CONCATENATE_AXIS)

	# 	# (4 x 4 x 8N)
	# 	dec2 = self._add_decoding_layer(filter_count, True, dec1)
	# 	dec2 = concatenate([dec2, enc5], axis=self.CONCATENATE_AXIS)

	# 	# (8 x 8 x 8N)
	# 	dec3 = self._add_decoding_layer(filter_count, True, dec2)
	# 	dec3 = concatenate([dec3, enc4], axis=self.CONCATENATE_AXIS)

	# 	# (16 x 16 x 4N)
	# 	filter_count = first_layer_filter_count*4
	# 	dec4 = self._add_decoding_layer(filter_count, False, dec3)
	# 	dec4 = concatenate([dec4, enc3], axis=self.CONCATENATE_AXIS)

	# 	# (32 x 32 x 2N)
	# 	filter_count = first_layer_filter_count*2
	# 	dec5 = self._add_decoding_layer(filter_count, False, dec4)
	# 	dec5 = concatenate([dec5, enc2], axis=self.CONCATENATE_AXIS)

	# 	# (64 x 64 x N)
	# 	filter_count = first_layer_filter_count
	# 	dec6 = self._add_decoding_layer(filter_count, False, dec5)
	# 	dec6 = concatenate([dec6, enc1], axis=self.CONCATENATE_AXIS)

	# 	# (256 x 256 x output_channel_count)
	# 	dec7 = Activation(activation='relu')(dec6)
	# 	dec7 = Conv2DTranspose(output_channel_count, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE)(dec7)
	# 	dec7 = Activation(activation='sigmoid')(dec7)

	# 	self.UNET = Model(inputs=inputs, outputs=dec7)
		
	#	U-net 512 : 9 layers
	# def __init__(self, input_channel_count, output_channel_count, first_layer_filter_count):
	# 	self.INPUT_IMAGE_SIZE = 512
	# 	self.CONCATENATE_AXIS = -1
	# 	self.CONV_FILTER_SIZE = 4
	# 	self.CONV_STRIDE = 2
	# 	self.CONV_PADDING = (1, 1)
	# 	self.DECONV_FILTER_SIZE = 2
	# 	self.DECONV_STRIDE = 2

	# 	# (512 x 512 x input_channel_count)
	# 	inputs = Input((self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, input_channel_count))

	# 	# エンコーダーの作成
	# 	# (256 x 256 x N)
	# 	enc1 = ZeroPadding2D(self.CONV_PADDING)(inputs)
	# 	enc1 = Conv2D(first_layer_filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(enc1)

	# 	# (128 x 128 x 2N)
	# 	filter_count = first_layer_filter_count*2
	# 	enc2 = self._add_encoding_layer(filter_count, enc1)

	# 	# (64 x 64 x 4N)
	# 	filter_count = first_layer_filter_count*4
	# 	enc3 = self._add_encoding_layer(filter_count, enc2)

	# 	# (32 x 32 x 4N)
	# 	enc4 = self._add_encoding_layer(filter_count, enc3)

	# 	# (16 x 16 x 8N)
	# 	filter_count = first_layer_filter_count*8
	# 	enc5 = self._add_encoding_layer(filter_count, enc4)

	# 	# (8 x 8 x 8N)
	# 	enc6 = self._add_encoding_layer(filter_count, enc5)

	# 	# (4 x 4 x 8N)
	# 	enc7 = self._add_encoding_layer(filter_count, enc6)

	# 	# (2 x 2 x 8N)
	# 	enc8 = self._add_encoding_layer(filter_count, enc7)

	# 	# (1 x 1 x 8N)
	# 	enc9 = self._add_encoding_layer(filter_count, enc8)

	# 	# デコーダーの作成
	# 	# (2 x 2 x 8N)
	# 	dec1 = self._add_decoding_layer(filter_count, True, enc9)
	# 	dec1 = concatenate([dec1, enc8], axis=self.CONCATENATE_AXIS)

	# 	# (4 x 4 x 8N)
	# 	dec2 = self._add_decoding_layer(filter_count, True, dec1)
	# 	dec2 = concatenate([dec2, enc7], axis=self.CONCATENATE_AXIS)

	# 	# (8 x 8 x 8N)
	# 	dec3 = self._add_decoding_layer(filter_count, True, dec2)
	# 	dec3 = concatenate([dec3, enc6], axis=self.CONCATENATE_AXIS)

	# 	# (16 x 16 x 8N)
	# 	dec4 = self._add_decoding_layer(filter_count, False, dec3)
	# 	dec4 = concatenate([dec4, enc5], axis=self.CONCATENATE_AXIS)

	# 	# (32 x 32 x 4N)
	# 	filter_count = first_layer_filter_count*4
	# 	dec5 = self._add_decoding_layer(filter_count, False, dec4)
	# 	dec5 = concatenate([dec5, enc4], axis=self.CONCATENATE_AXIS)

	# 	# (64 x 64 x 4N)
	# 	dec6 = self._add_decoding_layer(filter_count, False, dec5)
	# 	dec6 = concatenate([dec6, enc3], axis=self.CONCATENATE_AXIS)

	# 	# (128 x 128 x 2N)
	# 	filter_count = first_layer_filter_count*2
	# 	dec7 = self._add_decoding_layer(filter_count, False, dec6)
	# 	dec7 = concatenate([dec7, enc2], axis=self.CONCATENATE_AXIS)

	# 	# (128 x 128 x 2N)
	# 	dec8 = self._add_decoding_layer(filter_count, False, dec7)
	# 	dec8 = concatenate([dec8, enc1], axis=self.CONCATENATE_AXIS)

	# 	# (512 x 512 x output_channel_count)
	# 	filter_count = first_layer_filter_count
	# 	dec9 = Activation(activation='relu')(dec8)
	# 	dec9 = Conv2DTranspose(output_channel_count, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE)(dec9)
	# 	dec9 = Activation(activation='sigmoid')(dec9)

	# 	self.UNET = Model(inputs=inputs, outputs=dec9)


 	# # image size = 256 : 8 Layers
	# def __init__(self, input_channel_count, output_channel_count, first_layer_filter_count):
	# 	self.INPUT_IMAGE_SIZE = 256
	# 	self.CONCATENATE_AXIS = -1
	# 	self.CONV_FILTER_SIZE = 4
	# 	self.CONV_STRIDE = 2
	# 	self.CONV_PADDING = (1, 1)
	# 	self.DECONV_FILTER_SIZE = 2
	# 	self.DECONV_STRIDE = 2

	# 	# (256 x 256 x input_channel_count)
	# 	inputs = Input((self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, input_channel_count))

	# 	# エンコーダーの作成
	# 	# (128 x 128 x N)
	# 	enc1 = ZeroPadding2D(self.CONV_PADDING)(inputs)
	# 	enc1 = Conv2D(first_layer_filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(enc1)

	# 	# (64 x 64 x 2N)
	# 	filter_count = first_layer_filter_count*2
	# 	enc2 = self._add_encoding_layer(filter_count, enc1)

	# 	# (32 x 32 x 4N)
	# 	filter_count = first_layer_filter_count*4
	# 	enc3 = self._add_encoding_layer(filter_count, enc2)

	# 	# (16 x 16 x 8N)
	# 	filter_count = first_layer_filter_count*8
	# 	enc4 = self._add_encoding_layer(filter_count, enc3)

	# 	# (8 x 8 x 8N)
	# 	enc5 = self._add_encoding_layer(filter_count, enc4)

	# 	# (4 x 4 x 8N)
	# 	enc6 = self._add_encoding_layer(filter_count, enc5)

	# 	# (2 x 2 x 8N)
	# 	enc7 = self._add_encoding_layer(filter_count, enc6)

	# 	# (1 x 1 x 8N)
	# 	enc8 = self._add_encoding_layer(filter_count, enc7)

	# 	# デコーダーの作成
	# 	# (2 x 2 x 8N)
	# 	dec1 = self._add_decoding_layer(filter_count, True, enc8)
	# 	dec1 = concatenate([dec1, enc7], axis=self.CONCATENATE_AXIS)

	# 	# (4 x 4 x 8N)
	# 	dec2 = self._add_decoding_layer(filter_count, True, dec1)
	# 	dec2 = concatenate([dec2, enc6], axis=self.CONCATENATE_AXIS)

	# 	# (8 x 8 x 8N)
	# 	dec3 = self._add_decoding_layer(filter_count, True, dec2)
	# 	dec3 = concatenate([dec3, enc5], axis=self.CONCATENATE_AXIS)

	# 	# (16 x 16 x 8N)
	# 	dec4 = self._add_decoding_layer(filter_count, False, dec3)
	# 	dec4 = concatenate([dec4, enc4], axis=self.CONCATENATE_AXIS)

	# 	# (32 x 32 x 4N)
	# 	filter_count = first_layer_filter_count*4
	# 	dec5 = self._add_decoding_layer(filter_count, False, dec4)
	# 	dec5 = concatenate([dec5, enc3], axis=self.CONCATENATE_AXIS)

	# 	# (64 x 64 x 2N)
	# 	filter_count = first_layer_filter_count*2
	# 	dec6 = self._add_decoding_layer(filter_count, False, dec5)
	# 	dec6 = concatenate([dec6, enc2], axis=self.CONCATENATE_AXIS)

	# 	# (128 x 128 x N)
	# 	filter_count = first_layer_filter_count
	# 	dec7 = self._add_decoding_layer(filter_count, False, dec6)
	# 	dec7 = concatenate([dec7, enc1], axis=self.CONCATENATE_AXIS)

	# 	# (256 x 256 x output_channel_count)
	# 	dec8 = Activation(activation='relu')(dec7)
	# 	dec8 = Conv2DTranspose(output_channel_count, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE)(dec8)
	# 	dec8 = Activation(activation='sigmoid')(dec8)

	# 	self.UNET = Model(inputs=inputs, outputs=dec8)


	def _add_encoding_layer(self, filter_count, sequence):
		new_sequence = LeakyReLU(0.2)(sequence)
		new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
		new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
		new_sequence = BatchNormalization()(new_sequence)
		return new_sequence

	def _add_decoding_layer(self, filter_count, add_drop_layer, sequence):
		new_sequence = Activation(activation='relu')(sequence)
		new_sequence = Conv2DTranspose(filter_count, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE,
									   kernel_initializer='he_uniform')(new_sequence)
		new_sequence = BatchNormalization()(new_sequence)
		if add_drop_layer:
			new_sequence = Dropout(0.5)(new_sequence)
		return new_sequence

	def get_model(self):
		return self.UNET
