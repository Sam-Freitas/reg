from xml.etree.ElementInclude import include
# from tensorflow.keras.models import Model 
# from tensorflow.keras.layers import Input, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, BatchNormalization, Activation, Dense, LocallyConnected2D, SeparableConv2D
# from tensorflow.keras import backend as K
# from tensorflow.python.keras.engine import training
# from torch import channels_last, dropout
# from utils.DropBlock import DropBlock2D
from skimage import measure
from tqdm import tqdm
from natsort import natsorted
import matplotlib.pyplot as plt
import albumentations as A
import tensorflow as tf
import numpy as np
import imutils
import shutil
import random
import json
import glob
import sys
import cv2
import os

def fully_connected_CNN_v4(use_dropout = False, height = 128, width = 128, channels = 2, kernal_size = (3,3), 
    inital_filter_size = 16,keep_prob = 0.9,blocksize = 7, layers = 3, sub_layers = 3):

    inputs = Input((height, width, channels))

    s = inputs

    # first block of convolutions
    for i in range(layers):
        filt_mult = 2**i
        this_filter_size = inital_filter_size*filt_mult
        this_dropblock_size = int(round(inital_filter_size/filt_mult))
        if this_dropblock_size <= 0:
            this_dropblock_size = 1
        if i == 0:
            conv_3 = Conv2D(this_filter_size, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same', strides = (1,1))(s)
        else:
            conv_3 = Conv2D(inital_filter_size*filt_mult, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same', strides = (1,1))(pool_3)
        conv_3 = DropBlock2D(keep_prob = keep_prob, block_size = this_dropblock_size)(conv_3, training = use_dropout)
        conv_3 = Activation('elu')(conv_3)

        pool_3 = MaxPooling2D((2,2))(conv_3)

    flattened = tf.keras.layers.Flatten()(pool_3)

    d = Dense(512)(flattened)
    d = Activation('swish')(d)
    d = Dropout(0.75)(d, training = use_dropout)

    output = Dense(1,activation='linear')(d)

    model = Model(inputs=[inputs], outputs=[output])

    return model

def fully_connected_CNN_v3(use_dropout = False, height = 128, width = 128, channels = 2, kernal_size = (3,3), 
    inital_filter_size = 16,keep_prob = 0.9,blocksize = 7, layers = 3, sub_layers = 3):

    inputs = Input((height, width, channels))

    s = inputs

    # first block of convolutions
    for i in range(layers):
        filt_mult = 2**i
        this_filter_size = inital_filter_size*filt_mult
        this_dropblock_size = inital_filter_size/filt_mult
        if i == 0:
            conv_3 = Conv2D(this_filter_size, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same', strides = (1,1))(s)
        else:
            conv_3 = Conv2D(inital_filter_size*filt_mult, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same', strides = (1,1))(pool_3)
        conv_3 = DropBlock2D(keep_prob = keep_prob, block_size = blocksize)(conv_3, training = use_dropout)
        # conv_3 = BatchNormalization(momentum = 0.5)(conv_3)
        conv_3 = Activation('elu')(conv_3)


        for j in range(sub_layers):
            conv_3 = Conv2D(this_filter_size, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same', strides = (1,1))(conv_3)
            # conv_3 = DropBlock2D(keep_prob = keep_prob, block_size = blocksize)(conv_3, training = use_dropout)
            conv_3 = Activation('swish')(conv_3)

        pool_3 = MaxPooling2D((2,2))(conv_3)

    # pool_3 = DropBlock2D(keep_prob = keep_prob, block_size = blocksize)(pool_3, training = use_dropout)

    flattened = tf.keras.layers.Flatten()(pool_3)

    d = Dense(128)(flattened)
    d = Activation('swish')(d)
    d = Dropout(0.75)(d, training = use_dropout)

    output = Dense(1,activation='linear')(d)

    model = Model(inputs=[inputs], outputs=[output])

    return model

def fully_connected_CNN_v2(use_dropout = False, height = 128, width = 128, channels = 1, kernal_size = (3,3), inital_filter_size = 16,dropsize = 0.9,blocksize = 7):

    inputs = Input((height, width, channels))

    s = inputs

    for i in range(3):
        # first block of convolutions
        filt_mult = 2**i
        if i == 0:
            conv_1 = Conv2D(inital_filter_size*filt_mult, (15,15), activation = None, kernel_initializer = 'he_normal', padding = 'same', strides = (1,1))(s)
        else:
            conv_1 = Conv2D(inital_filter_size*filt_mult, (15,15), activation = None, kernel_initializer = 'he_normal', padding = 'same', strides = (1,1))(pool_1)
        conv_1 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_1, training = use_dropout)
        conv_1 = Activation('elu')(conv_1)

        pool_1 = MaxPooling2D((2,2))(conv_1)

    for i in range(3):
        # first block of convolutions
        filt_mult = 2**i
        if i == 0:
            conv_2 = Conv2D(inital_filter_size*filt_mult, (9,9), activation = None, kernel_initializer = 'he_normal', padding = 'same', strides = (1,1))(s)
        else:
            conv_2 = Conv2D(inital_filter_size*filt_mult, (9,9), activation = None, kernel_initializer = 'he_normal', padding = 'same', strides = (1,1))(pool_2)
        conv_2 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_2, training = use_dropout)
        conv_2 = Activation('elu')(conv_2)

        pool_2 = MaxPooling2D((2,2))(conv_2)

    # first block of convolutions
    for i in range(3):
        filt_mult = 2**i
        if i == 0:
            conv_3 = Conv2D(inital_filter_size*filt_mult, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same', strides = (1,1))(s)
        else:
            conv_3 = Conv2D(inital_filter_size*filt_mult, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same', strides = (1,1))(pool_3)
        conv_3 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_3, training = use_dropout)
        conv_3 = Activation('elu')(conv_3)

        pool_3 = MaxPooling2D((2,2))(conv_3)

    cat_layer = tf.keras.layers.Concatenate()([pool_1,pool_2,pool_3])

    flattened = tf.keras.layers.Flatten()(cat_layer)

    d = Dense(6000)(flattened)
    d = Activation('elu')(d)
    d = Dropout(0.1)(d, training = use_dropout)

    d_output = Dense(1,activation='linear')(d)

    inputs_metadata = Input(shape = (2,)) # sex, tissue type
    sm = Dense(512,input_shape = inputs_metadata.shape)(inputs_metadata)
    dm = Activation('elu')(sm)
    dm_output = Dense(1,activation='linear')(dm)

    #concatinations
    out_concat = tf.keras.layers.Add()([d_output,dm_output])

    # final output layes for data exportation
    output = Dense(1,activation='linear',name = 'continious_output')(out_concat)
    # output = tf.keras.layers.ReLU()(output)

    model = Model(inputs=[inputs,inputs_metadata], outputs=[output])

    return model

def fully_connected_CNN(use_dropout = False, height = 128, width = 128, channels = 1, kernal_size = (3,3), inital_filter_size = 16,dropsize = 0.9,blocksize = 7):

    inputs = Input((height, width, channels))

    s = inputs

    # # first block of convolutions
    # conv_1 = Conv2D(inital_filter_size, (1,1), activation = None, kernel_initializer = 'he_normal', padding = 'same')(s)
    # conv_1 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_1, training = use_dropout)
    # conv_1 = BatchNormalization()(conv_1)
    # conv_1 = Activation('relu')(conv_1)

    # conv_1 = Conv2D(inital_filter_size, (1,1), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_1)
    # conv_1 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_1, training = use_dropout)
    # conv_1 = BatchNormalization()(conv_1)
    # conv_1 = Activation('relu')(conv_1)

    # pool_1 = MaxPooling2D((2,2))(conv_1)

    # # second block of convolutions
    # conv_1 = Conv2D(inital_filter_size*2, (1,1), activation = None, kernel_initializer = 'he_normal', padding = 'same') (pool_1)
    # conv_1 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_1, training = use_dropout)
    # conv_1 = BatchNormalization()(conv_1)
    # conv_1 = Activation('relu')(conv_1)

    # conv_1 = Conv2D(inital_filter_size*2, (1,1), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_1)
    # conv_1 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_1, training = use_dropout)
    # conv_1 = BatchNormalization()(conv_1)
    # conv_1 = Activation('relu')(conv_1)

    # pool_1 = MaxPooling2D((2,2))(conv_1)

    # # third block of convolutions
    # conv_1 = Conv2D(inital_filter_size*4, (1,1), activation = None, kernel_initializer = 'he_normal', padding = 'same') (pool_1)
    # conv_1 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_1, training = use_dropout)
    # conv_1 = BatchNormalization()(conv_1)
    # conv_1 = Activation('relu')(conv_1)

    # conv_1 = Conv2D(inital_filter_size*4, (1,1), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_1)
    # conv_1 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_1, training = use_dropout)
    # conv_1 = BatchNormalization()(conv_1)
    # conv_1 = Activation('relu')(conv_1)

    # pool_1 = MaxPooling2D((2,2))(conv_1)

    # # first block of convolutions
    # conv_2 = Conv2D(inital_filter_size, (7,7), activation = None, kernel_initializer = 'he_normal', padding = 'same')(s)
    # conv_2 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_2, training = use_dropout)
    # conv_2 = BatchNormalization()(conv_2)
    # conv_2 = Activation('relu')(conv_2)

    # conv_2 = Conv2D(inital_filter_size, (7,7), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_2)
    # conv_2 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_2, training = use_dropout)
    # conv_2 = BatchNormalization()(conv_2)
    # conv_2 = Activation('relu')(conv_2)

    # pool_2 = MaxPooling2D((2,2))(conv_2)

    # # second block of convolutions
    # conv_2 = Conv2D(inital_filter_size*2, (7,7), activation = None, kernel_initializer = 'he_normal', padding = 'same') (pool_2)
    # conv_2 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_2, training = use_dropout)
    # conv_2 = BatchNormalization()(conv_2)
    # conv_2 = Activation('relu')(conv_2)

    # conv_2 = Conv2D(inital_filter_size*2, (7,7), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_2)
    # conv_2 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_2, training = use_dropout)
    # conv_2 = BatchNormalization()(conv_2)
    # conv_2 = Activation('relu')(conv_2)

    # pool_2 = MaxPooling2D((2,2))(conv_2)

    # # third block of convolutions
    # conv_2 = Conv2D(inital_filter_size*4, (7,7), activation = None, kernel_initializer = 'he_normal', padding = 'same') (pool_2)
    # conv_2 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_2, training = use_dropout)
    # conv_2 = BatchNormalization()(conv_2)
    # conv_2 = Activation('relu')(conv_2)

    # conv_2 = Conv2D(inital_filter_size*4, (7,7), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_2)
    # conv_2 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_2, training = use_dropout)
    # conv_2 = BatchNormalization()(conv_2)
    # conv_2 = Activation('relu')(conv_2)

    # pool_2 = MaxPooling2D((2,2))(conv_2)

    # first block of convolutions
    conv_3 = Conv2D(inital_filter_size, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same')(s)
    conv_3 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_3, training = use_dropout)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('relu')(conv_3)

    conv_3 = Conv2D(inital_filter_size, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_3)
    conv_3 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_3, training = use_dropout)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('relu')(conv_3)

    pool_3 = MaxPooling2D((2,2))(conv_3)

    # second block of convolutions
    conv_3 = Conv2D(inital_filter_size*2, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same') (pool_3)
    conv_3 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_3, training = use_dropout)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('relu')(conv_3)

    conv_3 = Conv2D(inital_filter_size*2, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_3)
    conv_3 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_3, training = use_dropout)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('relu')(conv_3)

    pool_3 = MaxPooling2D((2,2))(conv_3)

    # third block of convolutions
    conv_3 = Conv2D(inital_filter_size*4, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same') (pool_3)
    conv_3 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_3, training = use_dropout)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('relu')(conv_3)

    conv_3 = Conv2D(inital_filter_size*4, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_3)
    conv_3 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_3, training = use_dropout)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('relu')(conv_3)

    pool_3 = MaxPooling2D((2,2))(conv_3)

    # # first block of convolutions
    # conv_4 = Conv2D(inital_filter_size, (11,11), activation = None, kernel_initializer = 'he_normal', padding = 'same')(s)
    # conv_4 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_4, training = use_dropout)
    # conv_4 = BatchNormalization()(conv_4)
    # conv_4 = Activation('relu')(conv_4)

    # conv_4 = Conv2D(inital_filter_size, (11,11), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_4)
    # conv_4 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_4, training = use_dropout)
    # conv_4 = BatchNormalization()(conv_4)
    # conv_4 = Activation('relu')(conv_4)

    # pool_4 = MaxPooling2D((2,2))(conv_4)

    # # second block of convolutions
    # conv_4 = Conv2D(inital_filter_size*2, (11,11), activation = None, kernel_initializer = 'he_normal', padding = 'same') (pool_4)
    # conv_4 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_4, training = use_dropout)
    # conv_4 = BatchNormalization()(conv_4)
    # conv_4 = Activation('relu')(conv_4)

    # conv_4 = Conv2D(inital_filter_size*2, (11,11), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_4)
    # conv_4 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_4, training = use_dropout)
    # conv_4 = BatchNormalization()(conv_4)
    # conv_4 = Activation('relu')(conv_4)

    # pool_4 = MaxPooling2D((2,2))(conv_4)

    # # third block of convolutions
    # conv_4 = Conv2D(inital_filter_size*4, (11,11), activation = None, kernel_initializer = 'he_normal', padding = 'same') (pool_4)
    # conv_4 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_4, training = use_dropout)
    # conv_4 = BatchNormalization()(conv_4)
    # conv_4 = Activation('relu')(conv_4)

    # conv_4 = Conv2D(inital_filter_size*4, (11,11), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_4)
    # conv_4 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_4, training = use_dropout)
    # conv_4 = BatchNormalization()(conv_4)
    # conv_4 = Activation('relu')(conv_4)

    # pool_4 = MaxPooling2D((2,2))(conv_4)

    # to_dense = tf.keras.layers.Concatenate()([pool_1,pool_2,pool_3,pool_4])

    flattened = tf.keras.layers.Flatten()(pool_3)
    
    d = Dense(1024,activation='relu')(flattened)

    d = Dropout(0.3)(d, training = use_dropout)

    d = Dense(128,activation='relu')(flattened)

    d = Dropout(0.3)(d, training = use_dropout)

    output = Dense(1,activation='linear')(d)

    model = Model(inputs=[inputs], outputs=[output])

    return model

def ResNet50v2_regression(use_dropout = False, height = 128, width = 128, channels = 1):

    base_model = tf.keras.applications.resnet_v2.ResNet50V2(
        include_top = False, weights = None, input_shape = (height,width,channels)
    )
    last_base_layer = base_model.get_layer('post_bn').output
    x = tf.keras.layers.Flatten()(last_base_layer)
    x = Dense(1024,activation='relu')(x)
    x = Dropout(0.5)(x, training = use_dropout)
    x = Dense(1,activation='linear')(x)

    model = Model(inputs = base_model.input, outputs = x)


    return model

def plot_model(model):

    try:
        tf.keras.utils.plot_model(
            model, to_file='model.png', show_shapes=True, show_dtype=True,
            show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)
    except:
        print("Exporting model to png failed")
        print("Necessary packages: pydot (pip) and graphviz (brew)")

def load_rotated_minst_dataset(seed = None):

    (X,labels),(_,_) = tf.keras.datasets.mnist.load_data()

    only_1 = [labels==1][0]

    labels = labels[only_1]
    X = X[only_1]

    X = X[3]

    X_out = []
    y_out = []

    if seed is not None:
        np.random.seed(seed)

    y_out = np.asarray(np.random.randint(low = -90, high = 90, size = 1500))

    for count,this_angle in enumerate(y_out):
        rot = imutils.rotate(X, angle=this_angle)
        X_out.append(rot)

    X_out = np.asarray(X_out)
    y_out = np.asarray(y_out)

    X_out_test = X_out[-200:]
    y_out_test = y_out[-200:]

    X_out_val = X_out[-500:-200]
    y_out_val = y_out[-500:-200]

    X_out = X_out[:1000]
    y_out = y_out[:1000]

    return (X_out,y_out),(X_out_val,y_out_val) ,(X_out_test,y_out_test)

def diff_func(X_norm,y_norm,age_normalizer = 1):
    print('Diff function generation')
    y_diff = []
    X_diff = []
    num_loops = y_norm.shape[0]
    count = 0
    for i in tqdm(range(num_loops)):
        X1 = X_norm[i]
        y1 = y_norm[i]
        for j in range(num_loops):
            X2 = X_norm[j]
            y2 = y_norm[j]
            X_diff.append(np.concatenate([np.atleast_3d(X1),np.atleast_3d(X2)],axis = -1).squeeze())
            y_temp = (y1-y2)/age_normalizer
            y_temp = np.round(y_temp,3)
            y_diff.append(y_temp)
            count = count + 1
    X_diff = np.asarray(X_diff)
    y_diff = np.asarray(y_diff)

    return X_diff, y_diff

class test_on_improved_val_loss(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):

        curr_path = os.path.split(__file__)[0]

        curr_val_loss = logs['val_loss']
        try:
            val_loss_hist = self.model.history.history['val_loss']
        except:
            val_loss_hist = curr_val_loss + 1

        if epoch == 0:
            try:
                os.mkdir(os.path.join(curr_path, 'output_images_testing_during'))
            except:
                shutil.rmtree(os.path.join(curr_path, 'output_images_testing_during'))
                os.mkdir(os.path.join(curr_path, 'output_images_testing_during'))

        if curr_val_loss < np.min(val_loss_hist) or epoch == 0:
            print("val_loss improved to:",curr_val_loss)
            loss_flag = True
        else:
            print("Earlystop:,", epoch - np.argmin(val_loss_hist))
            loss_flag = False

        if (epoch % 10) == 0 or loss_flag:

            print('Testing on epoch', str(epoch))

            temp = np.load(os.path.join(curr_path,'data_arrays','test.npz'))
            X_test,X_meta_test,y_test = temp['X'],temp['X_meta'],temp['y']

            eval_result = self.model.evaluate([X_test,X_meta_test],[y_test],batch_size=1,verbose=0,return_dict=True)
            print(eval_result)

            # plt.figure(1)
            plt.close('all')

            predicted = []
            for i in range(5):
                predicted.append(self.model.predict([X_test,X_meta_test],batch_size=1).squeeze())
            predicted = np.asarray(predicted)
            predicted = np.mean(predicted,axis = 0)

            cor_matrix = np.corrcoef(predicted,y_test)
            cor_xy = cor_matrix[0,1]
            r_squared = round(cor_xy**2,4)
            print("Current r_squared test:",r_squared)

            res = dict()
            for key in eval_result: res[key] = round(eval_result[key],6)

            plt.scatter(y_test,predicted,color = 'r',alpha=0.2)
            plt.plot(np.linspace(np.min(y_test), np.max(y_test)),np.linspace(np.min(y_test), np.max(y_test)))
            plt.text(np.min(y_test),np.max(y_test),"r^2: " + str(r_squared),fontsize = 12)
            plt.title(json.dumps(res).replace(',', '\n'),fontsize = 10)
            plt.xlabel('Expected Age (years)')
            plt.ylabel('Predicted Age (years)')

            if loss_flag:
                extn = 'best_'
            else:
                extn = ''

            output_name = os.path.join(curr_path,'output_images_testing_during',str(epoch) + '_' + str(r_squared)[2:] + extn +'.png')

            plt.savefig(fname = output_name)

            plt.close('all')


class test_on_improved_val_lossv3(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):

        plt.ioff()

        curr_path = os.path.split(__file__)[0]

        curr_val_loss = logs['val_loss']
        try:
            val_loss_hist = self.model.history.history['val_loss']
        except:
            val_loss_hist = curr_val_loss + 1

        path_to_out = os.path.join(curr_path, 'output_images_testing_during')

        if epoch == 0:
            try:
                os.mkdir(path_to_out)
            except:
                shutil.rmtree(path_to_out)
                os.mkdir(path_to_out)

        if not self.model.history.epoch:
            num_k_fold = len(glob.glob(os.path.join(path_to_out,'*/')))
            os.mkdir(os.path.join(path_to_out,str(num_k_fold)))
            print('Kfold -',num_k_fold)
        else:
            num_k_fold = len(glob.glob(os.path.join(path_to_out,'*/')))-1
            print('Kfold -',num_k_fold)

        if curr_val_loss < np.min(val_loss_hist) or epoch == 0:
            loss_flag = True
        else:
            print("Earlystop:,", epoch - self.model.history.epoch[0] - np.argmin(val_loss_hist))
            loss_flag = False

        if loss_flag: #(epoch % 100) == 0 or loss_flag: 

            print('Testing on epoch', str(epoch))

            temp = np.load(os.path.join(curr_path,'data_arrays','test.npz'))
            X_test,y_test = temp['X'],temp['y']

            eval_result = self.model.evaluate([X_test],[y_test],batch_size=1,verbose=0,return_dict=True)
            print(eval_result)

            # plt.figure(1)
            plt.close('all')

            predicted = []
            for i in range(5):
                predicted.append(self.model.predict([X_test],batch_size=1).squeeze())
            predicted = np.asarray(predicted)
            predicted = np.mean(predicted,axis = 0)

            cor_matrix = np.corrcoef(predicted,y_test)
            cor_xy = cor_matrix[0,1]
            r_squared = round(cor_xy**2,4)
            print("Current r_squared test:",r_squared)

            res = dict()
            m, b = np.polyfit(y_test,predicted, deg = 1)
            res['b'] = b
            res['m'] = m
            for key in eval_result: res[key] = round(eval_result[key],6)

            plt.ioff()
            plt.scatter(y_test,predicted,color = 'r',alpha=0.2)
            plt.plot(np.linspace(np.min(y_test), np.max(y_test)),np.linspace(np.min(y_test), np.max(y_test)))
            plt.text(np.min(y_test),np.max(y_test),"r^2: " + str(r_squared),fontsize = 12)
            plt.title(json.dumps(res).replace(',', '\n'),fontsize = 8)
            plt.xlabel('Expected Age (years)')
            plt.ylabel('Predicted Age (years)')
            plt.plot(y_test, m*y_test + b,'m-')

            if loss_flag:
                extn = 'best_'
            else:
                extn = ''

            output_name = os.path.join(curr_path,'output_images_testing_during',str(num_k_fold),str(epoch) + '_' + str(r_squared)[2:] + extn +'.png')

            plt.savefig(fname = output_name)

            plt.close('all')