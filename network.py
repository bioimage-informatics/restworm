# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 11:11:33 2018

@author: Administrator
"""

from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, ZeroPadding2D, MaxPooling2D, UpSampling2D
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import LeakyReLU

class RESTWORM_NET(object):
    def __init__(self, input_image_size, input_channel_count, output_channel_count, first_layer_filter_count, num_layers, act, maxPool):
        

        self.input_image_size = input_image_size
        self.CONCATENATE_AXIS = -1
        if maxPool:
            self.CONV_FILTER_SIZE = 3
            self.CONV_STRIDE = 1            
        else:
            self.CONV_FILTER_SIZE = 4
            self.CONV_STRIDE = 2
        self.CONV_PADDING = (1, 1)
        self.DECONV_FILTER_SIZE = 2
        self.DECONV_STRIDE = 2
        self.input_channel_count = input_channel_count
        self.output_channel_count = output_channel_count
        self.first_layer_filter_count = first_layer_filter_count
        self.num_layers = num_layers
        self.act = act
        self.maxPool = maxPool

        # (512 x 512 x input_channel_count)
        ipts = Input((self.input_image_size, self.input_image_size, self.input_channel_count))
        print("input size: ", ipts.shape)
        currentLayer = ipts
        
        # エンコーダーの作成
        # (512 x 512 x N)
        if self.num_layers >= 1:
            filter_count = self.first_layer_filter_count
            new_sequence = ZeroPadding2D(self.CONV_PADDING)(ipts)
            new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
            new_sequence = LeakyReLU()(new_sequence)
            new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
            new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
            new_sequence = LeakyReLU()(new_sequence)
            enc1 = new_sequence
            print("enc1 size: ", enc1.shape)
            currentLayer = enc1

        # (256 x 256 x 2N)
        if self.num_layers >= 2:
            new_sequence = MaxPooling2D(pool_size=(2, 2))(currentLayer)
            new_sequence = BatchNormalization()(new_sequence)
            filter_count = self.first_layer_filter_count*2
            new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
            new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
            new_sequence = LeakyReLU()(new_sequence)
            new_sequence = BatchNormalization()(new_sequence)
            new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
            new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
            new_sequence = LeakyReLU()(new_sequence)
            enc2 = new_sequence
            print("enc2 size: ", enc2.shape)
            currentLayer = enc2
            
        # (128 x 128 x 4N)
        if self.num_layers >= 3:
            new_sequence = MaxPooling2D(pool_size=(2, 2))(currentLayer)
            new_sequence = BatchNormalization()(new_sequence)
            filter_count = self.first_layer_filter_count*4
            new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
            new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
            new_sequence = LeakyReLU()(new_sequence)
            new_sequence = BatchNormalization()(new_sequence)
            new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
            new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
            new_sequence = LeakyReLU()(new_sequence)
            enc3 = new_sequence
            print("enc3 size: ", enc3.shape)
            currentLayer = enc3
            
        # (64 x 64 x 8N)
        if self.num_layers >= 4:
            new_sequence = MaxPooling2D(pool_size=(2, 2))(currentLayer)
            new_sequence = BatchNormalization()(new_sequence)
            filter_count = self.first_layer_filter_count*8
            new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
            new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
            new_sequence = LeakyReLU()(new_sequence)
            new_sequence = BatchNormalization()(new_sequence)
            new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
            new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
            new_sequence = LeakyReLU()(new_sequence)
            enc4 = new_sequence
            print("enc4 size: ", enc4.shape)
            currentLayer = enc4
            
        # (32 x 32 x 16N)
        if self.num_layers >= 5:
            new_sequence = MaxPooling2D(pool_size=(2, 2))(currentLayer)
            new_sequence = BatchNormalization()(new_sequence)
            filter_count = self.first_layer_filter_count*16
            new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
            new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
            new_sequence = LeakyReLU()(new_sequence)
            new_sequence = BatchNormalization()(new_sequence)
            new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
            new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
            new_sequence = LeakyReLU()(new_sequence)
            enc5 = new_sequence
            print("enc5 size: ", enc5.shape)
            currentLayer = enc5
            
        # (16 x 16 x 16N)
        if self.num_layers >= 6:
            new_sequence = MaxPooling2D(pool_size=(2, 2))(currentLayer)
            new_sequence = BatchNormalization()(new_sequence)
            filter_count = self.first_layer_filter_count*16
            new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
            new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
            new_sequence = LeakyReLU()(new_sequence)
            new_sequence = BatchNormalization()(new_sequence)
            new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
            new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
            new_sequence = LeakyReLU()(new_sequence)
            enc6 = new_sequence
            print("enc6 size: ", enc6.shape)
            currentLayer = enc6
    
        # (256 x 256 x N)
        if self.num_layers >= 6:
            new_sequence = UpSampling2D(size = (2,2))(currentLayer)
            new_sequence = concatenate([new_sequence, enc5], axis=self.CONCATENATE_AXIS)
            new_sequence = BatchNormalization()(new_sequence)
            filter_count = self.first_layer_filter_count*16
            new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
            new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
            new_sequence = Activation(activation='relu')(new_sequence)
            new_sequence = BatchNormalization()(new_sequence)
            new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
            new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
            new_sequence = Activation(activation='relu')(new_sequence)
            dec6 = new_sequence
            print("dec6 size: ", dec6.shape)
            currentLayer = dec6

        # (256 x 256 x N)
        if self.num_layers >= 5:
            new_sequence = UpSampling2D(size = (2,2))(currentLayer)
            new_sequence = concatenate([new_sequence, enc4], axis=self.CONCATENATE_AXIS)
            new_sequence = BatchNormalization()(new_sequence)
            filter_count = self.first_layer_filter_count*8
            new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
            new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
            new_sequence = Activation(activation='relu')(new_sequence)
            new_sequence = BatchNormalization()(new_sequence)
            new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
            new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
            new_sequence = Activation(activation='relu')(new_sequence)
            dec5 = new_sequence
            print("dec5 size: ", dec5.shape)
            currentLayer = dec5

        # (256 x 256 x N)
        if self.num_layers >= 4:
            new_sequence = UpSampling2D(size = (2,2))(currentLayer)
            new_sequence = concatenate([new_sequence, enc3], axis=self.CONCATENATE_AXIS)
            new_sequence = BatchNormalization()(new_sequence)
            filter_count = self.first_layer_filter_count*4
            new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
            new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
            new_sequence = Activation(activation='relu')(new_sequence)
            new_sequence = BatchNormalization()(new_sequence)
            new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
            new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
            new_sequence = Activation(activation='relu')(new_sequence)
            dec4 = new_sequence
            print("dec4 size: ", dec4.shape)
            currentLayer = dec4

        # (256 x 256 x N)
        if self.num_layers >= 3:
            new_sequence = UpSampling2D(size = (2,2))(currentLayer)
            new_sequence = concatenate([new_sequence, enc2], axis=self.CONCATENATE_AXIS)
            new_sequence = BatchNormalization()(new_sequence)
            filter_count = self.first_layer_filter_count*2
            new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
            new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
            new_sequence = Activation(activation='relu')(new_sequence)
            new_sequence = BatchNormalization()(new_sequence)
            new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
            new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
            new_sequence = Activation(activation='relu')(new_sequence)
            dec3 = new_sequence
            print("dec3 size: ", dec3.shape)
            currentLayer = dec3

        # (256 x 256 x N)
        if self.num_layers >= 2:
            new_sequence = UpSampling2D(size = (2,2))(currentLayer)
            new_sequence = concatenate([new_sequence, enc1], axis=self.CONCATENATE_AXIS)
            new_sequence = BatchNormalization()(new_sequence)
            filter_count = self.first_layer_filter_count
            new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
            new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
            new_sequence = Activation(activation='relu')(new_sequence)
            new_sequence = BatchNormalization()(new_sequence)
            new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
            new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
            new_sequence = Activation(activation='relu')(new_sequence)
            dec2 = new_sequence
            print("dec2 size: ", dec2.shape)
            currentLayer = dec2
            # アウトプット
            new_sequence = ZeroPadding2D(self.CONV_PADDING)(currentLayer)
            dec1 = Conv2D(self.output_channel_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
            print("dec1 size: ", dec1.shape)
            
        self.UNET = Model(inputs=ipts, outputs=dec1)

    def get_model(self):
        return self.UNET
    
