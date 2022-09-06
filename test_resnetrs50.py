import math
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import imutils 
import cv2
from utils import load_rotated_minst_dataset, load_GTEx

class test_on_imporved_val(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = None):
        curr_val_loss = logs['val_loss']
        try:
            val_loss_hist = self.model.history.history['val_loss']
        except:
            val_loss_hist = curr_val_loss + 1
        
        if curr_val_loss < np.min(val_loss_hist):

            print('\nImproved val loss')

# Setting seed for reproducibiltiy
SEED = 50
tf.keras.utils.set_random_seed(SEED)

# (X_train,y_train),(X_val,y_val) ,(X_test,y_test) = load_rotated_minst_dataset(
#     seed = SEED, img_width=224,img_height=224,img_channels = 3 ,num = 1500)

X_train, y_train, X_test, y_test = load_GTEx(
    '/Users/samfreitas/Documents/Sutphin lab/Neural Nets/IGTD/Results/GTEx_merge_L1000_spiral_rgb',
    '/Users/samfreitas/Documents/Sutphin lab/Neural Nets/IGTD/Data/GTEx_merge_L1000_sample.csv',
    (224,224,3), n = 100, seed=SEED
    )

base_model = tf.keras.applications.resnet_rs.ResNetRS50(
    include_top = False, include_preprocessing = True)

base_model = tf.keras.applications.resnet_v2.ResNet50V2(include_top = False)

# for layer in base_model.layers:
#     layer.trainable = False
training = True

x = base_model.output
# x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512,activation = 'relu')(x)
x = tf.keras.layers.Dense(256,activation = 'relu')(x)
x = tf.keras.layers.Dense(128,activation = 'relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
model_outputs = tf.keras.layers.Dense(1,activation = 'linear')(x)
model = tf.keras.Model(inputs = base_model.input, outputs = model_outputs)

learning_rate = 0.001

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
    loss = tf.keras.losses.MeanAbsoluteError(), 
    metrics = tfa.metrics.RSquare(),
)
# model.compile(optimizer = optimizer,
#     loss = tf.keras.losses.MeanAbsoluteError(), 
#     metrics = tfa.metrics.RSquare()
# )

# tf.keras.utils.plot_model(
#     model,
#     to_file='model.png',
#     show_shapes=False,
#     show_dtype=False,
#     show_layer_names=True,
#     rankdir='TB',
#     expand_nested=False,
#     dpi=96,
#     layer_range=None,
#     show_layer_activations=False
# )

model.summary()

epochs = 100
check_loss = test_on_imporved_val()
es = tf.keras.callbacks.EarlyStopping(restore_best_weights = True, patience = epochs)

history = model.fit(
    x = X_train,
    y = y_train,
    validation_data = [X_test,y_test],
    epochs = epochs,
    callbacks = [check_loss,es]
)

for layer in model.layers:
    if layer.name == 'dropout':
        layer.rate = 0.0

model.save_weights('model_weights/')

predictions = model.predict(X_test)
evaluations = model.evaluate(x = X_test,y = y_test, return_dict = True)
predictions_on_train = model.predict(X_train)

mae = tf.keras.losses.MeanAbsoluteError()
test_loss = mae(y_test, predictions).numpy()

print(test_loss)

plt.figure(1)
plt.plot(y_test,y_test,'b-')
plt.plot(y_train,predictions_on_train,'bo')
plt.plot(y_test,predictions,'ro')
plt.show()

print('eof')