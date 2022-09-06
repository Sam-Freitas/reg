from importlib.metadata import metadata
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
import cv2 
import imutils
import pandas as pd
import glob 
import os
import pathlib
from natsort import natsorted, natsort_keygen

def load_GTEx(path_to_img_folder,path_to_metadata_csv,img_size,seed = 50,n=100, source = None):

    # img_size = (224,224,3)

    print('Loading in GTEx data as Images')
    imgs_df = pd.DataFrame()
    imgs_df['ID'] = glob.glob(os.path.join(path_to_img_folder,'*.png'))
    metadata = pd.read_csv(path_to_metadata_csv)

    imgs_df = imgs_df.sort_values(by= "ID", key=natsort_keygen())
    imgs_df = imgs_df.reset_index(drop=True)
    metadata = metadata.sort_values(by= "ID", key=natsort_keygen())
    metadata = metadata.reset_index(drop=True)

    assert len(imgs_df) == len(metadata)

    test_img = cv2.imread(imgs_df['ID'].values[0])
    img_size = list(img_size)
    img_size.insert(0,len(imgs_df))

    X = np.zeros(shape = img_size,dtype = test_img.dtype)
    y = np.zeros(shape = (len(imgs_df),))

    i = 0
    for count,val in imgs_df.iterrows():
        this_metadata = metadata.iloc[count]

        if this_metadata.Source == 'Shokhirev':
            temp_img = cv2.imread(val["ID"])
            temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)

            temp_img = cv2.resize(temp_img,(img_size[1],img_size[2]))
            X[i] = temp_img

            y[i] = this_metadata['Age']

            i = i + 1
    
    X = X[:i]
    y = y[:i]

    # X = X.astype(np.float64)
    # X = 2*((X - X.min()) / (X.max() - X.min())) - 1

    idx = np.arange(len(y))
    np.random.seed(seed)
    test_idx = np.unique(np.random.randint(low = 0, high= len(y),size = (n,1)))
    temp = idx[test_idx]
    idx = np.delete(idx,test_idx)
    test_idx = temp

    X_test,y_test = X[test_idx],y[test_idx]
    X_out, y_out = X[idx],y[idx]

    return X_out, y_out, X_test, y_test


def load_rotated_minst_dataset(seed = None, img_width = None,img_height = None, num = 1500, img_channels = None):

    print('Loading in rotated dataset')

    (X,labels),(_,_) = tf.keras.datasets.mnist.load_data()

    only_1 = [labels==1][0]
    labels = labels[only_1]
    X = X[only_1]

    X = X[3]

    X_out = []
    y_out = []

    if seed is not None:
        np.random.seed(seed)

    if img_width == None or img_height == None:
        img_height = img_width = X.shape[1]

    y_out = np.asarray(np.random.randint(low = -90, high = 90, size = num))

    for count,this_angle in enumerate(y_out):
        rot = imutils.rotate(X, angle=this_angle)
        rot = imutils.resize(rot,img_width,img_height)
        if img_channels is not None:
            if img_channels == 1:
                rot = np.expand_dims(rot, axis = -1)
            if img_channels == 3:
                rot = cv2.cvtColor(rot,cv2.COLOR_GRAY2RGB)

        X_out.append(rot)

    X_out = np.asarray(X_out)
    y_out = np.asarray(y_out)

    indexs = np.round(np.linspace(1,num,11)).astype(np.int)[:-1]

    # 70% train 20% val 10% test

    X_out_test = X_out[indexs[-1]:]
    y_out_test = y_out[indexs[-1]:]

    X_out_val = X_out[indexs[-3]:indexs[-1]]
    y_out_val = y_out[indexs[-3]:indexs[-1]]

    X_out = X_out[:indexs[-3]]
    y_out = y_out[:indexs[-3]]

    return (X_out,y_out),(X_out_val,y_out_val) ,(X_out_test,y_out_test)