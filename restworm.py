# -*- coding: utf-8 -*-
from keras.utils import multi_gpu_model
import os, cv2
import tifffile as tiff
import numpy as np
import keras.callbacks
from network import RESTWORM_NET
import glob

# Folders
train_input_folder = "E:\Azuma\Restoration\Data\Training\Input"
train_gt_folder = "E:\Azuma\Restoration\Data\Training\GroundTruth"
test_input_folder = "E:\Azuma\Restoration\Data\Test\Input"
test_pred_folder = "E:\Azuma\Restoration\Data\Test\Prediction"

# model architecture
input_image_size = 512
input_channel_count = 1
output_channel_count = 1
first_layer_filter_count = 64

# Early stopping
es='No'
# data augmentation
aug='No'
# Loss function
loss = 'mse'
# Computation size parameters
BATCH_SIZE = 4
NUM_EPOCH = 5
num_layers = 5
maxPool = True
act = 'None'
ROW_SIZE = 512
COLUME_SIZE = 512

# load network
network = RESTWORM_NET(input_image_size, input_channel_count, output_channel_count, first_layer_filter_count, num_layers, act, maxPool)
input_image_size
# model
model = network.get_model()
# when using multi gpu
parallel_model = multi_gpu_model(model, gpus=2)
# compile
parallel_model.compile(optimizer='adam', loss=loss)

# Read training data
paths = glob.glob(train_input_folder + "\*.tif")
x_train = np.zeros((len(paths), COLUME_SIZE, ROW_SIZE, 1), np.float32)
for i, image_file in enumerate(paths):
    # x_train
    img = cv2.imread(image_file, -1)
    x_train[i,:,:,0] = img

paths = glob.glob(train_gt_folder + "\*.tif")
y_train = np.zeros((len(paths), COLUME_SIZE, ROW_SIZE, 1), np.float32)
for i, image_file in enumerate(paths):
    # y_train
    img = cv2.imread(image_file, -1)
    y_train[i,:,:,0] = img

# Training
if es=='Yes':
    es_cb = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=0)]
    test_ratio = 0.1
elif es=='No':
    es_cb = None
    test_ratio = None

parallel_model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCH, verbose=1, callbacks=es_cb, validation_split=test_ratio )


# Read test data
os.makedirs(test_pred_folder, exist_ok=True)
paths = glob.glob(test_input_folder + "\*.tif")
x_test = np.zeros((len(paths), COLUME_SIZE, ROW_SIZE, 1), np.float32)
for i, image_file in enumerate(paths):
    img = cv2.imread(image_file, -1)
    x_test[i,:,:,0] = img
        
# prediction
y_pred = parallel_model.predict(x_test, BATCH_SIZE)

# Save
for i, y in enumerate(y_pred):
    filename = os.path.split(paths[i])[1]
    this_name = test_pred_folder + os.sep + filename
    tiff.imsave(this_name, y)

