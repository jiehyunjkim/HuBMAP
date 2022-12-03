import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
from PIL import Image
import random
import pandas as pd
import cv2


DATASET_PATH = "/Users/jiehyun/kaggle/"
TRAIN_CSV = DATASET_PATH + "input/hubmap-organ-segmentation/train.csv"
train_df = pd.read_csv(TRAIN_CSV)
#OUTPUT_FOLDER = "/Users/jiehyun/kaggle/output/"


class unetwrapper:
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    def convert_to_npy(self):
        #ref: https://gist.github.com/anilsathyan7/ffb35601483ac46bd72790fde55f5c04
        dirs = os.listdir(self.data_dir)
        dirs.sort()
        lists = []
        for item in dirs:
            if os.path.isfile(self.data_dir+item):
                im = Image.open(self.data_dir+item).convert("RGB")
                im = np.array(im)
                lists.append(im)
        imgset=np.array(lists)
        #np.save("imgds.npy",imgset)

    def load_npy(self):
        loadedimages = []
        loadedmasks = []

        for i in range(len(train_df['id'])):
            idx = random.randint(0, len(train_df) - 1)
            img_id = train_df['id'][idx]
            loadedimages += [np.load(self.data_dir + f'img_npy_512/{img_id}.npy', allow_pickle=True).copy()]
            loadedmasks += [np.load(self.data_dir + f'mask_npy_512/{img_id}.npy', allow_pickle=True).copy()]

        imgs = np.asarray(loadedimages)
        masks = np.asarray(loadedmasks)

        return imgs, masks

    def change_demension(self):
        imgs, masks = self.load_npy()
        x = np.asarray(imgs,dtype=np.float32)/255
        y = np.asarray(masks,dtype=np.float32)/255
        y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)

        return x, y

    def test_train_split(self):
        from sklearn.model_selection import train_test_split
        x, y = self.change_demension()
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.145, random_state=0)

        #print("x_train: ", x_train.shape)
        #print("y_train: ", y_train.shape)
        #print("x_val: ", x_val.shape)
        #print("y_val: ", y_val.shape)

        return x_train, x_val, y_train, y_val

    def data_augmentation(self):
        from keras_unet.utils import get_augmented
        x_train, x_val, y_train, y_val = self.test_train_split()
        train_gen = get_augmented(
            x_train, y_train, batch_size=2,
            data_gen_args = dict(
                rotation_range=5.,
                width_shift_range=0.05,
                height_shift_range=0.05,
                shear_range=40,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=False,
                fill_mode='constant'
            ))
        
        return train_gen
            
    def unet(self):
        from keras_unet.models import custom_unet
        x_train, x_val, y_train, y_val = self.test_train_split()
        input_shape = x_train[0].shape

        model = custom_unet(
            input_shape,
            filters=32,
            use_batch_norm=True,
            dropout=0.3,
            dropout_change_per_layer=0.0,
            num_layers=5
        )

        return model

    def check_point(self):
        from keras.callbacks import ModelCheckpoint

        model_filename = 'segm_model_v3.h5'
        callback_checkpoint = ModelCheckpoint(
            model_filename, 
            verbose=1, 
            monitor='val_loss', 
            save_best_only=True,
        )

        return callback_checkpoint

    def compile(self):
        from keras.optimizers import Adam
        from keras_unet.metrics import iou, iou_thresholded, dice_coef

        model = self.unet()
        model.compile(
            optimizer=Adam(), 
            loss='binary_crossentropy',
            metrics=[iou, iou_thresholded, dice_coef, 'accuracy', 'sparse_categorical_accuracy']
        )

        train_gen = self.data_augmentation()
        callback_checkpoint = self.check_point()
        x_train, x_val, y_train, y_val = self.test_train_split()

        history = model.fit(
        train_gen,
        steps_per_epoch=10,
        epochs=5,
        
        validation_data=(x_val, y_val),
        callbacks=[callback_checkpoint]
        )

        return history

    '''
    def fit(self):
        model = self.unet()
        train_gen = self.data_augmentation()
        callback_checkpoint = self.check_point()
        x_train, x_val, y_train, y_val = self.test_train_split()

        history = model.fit(
        train_gen,
        steps_per_epoch=100,
        epochs=50,
        
        validation_data=(x_val, y_val),
        callbacks=[callback_checkpoint]
        )

        return history
    '''

    def predict(self):
        model = self.unet()
        model_filename = 'segm_model_v3.h5'
        x_train, x_val, y_train, y_val = self.test_train_split()

        model.load_weights(model_filename)
        y_pred = model.predict(x_val)
        return y_pred

    def threshold(self):
        y_pred = self.predict()

        a = y_pred
        a_binary = np.zeros(a.shape, dtype=np.bool)
        a_binary[a > 0.1] = True

        return a_binary

    def plot_resuts(self):
        from keras_unet.utils import plot_imgs
        a_binary = self.threshold()

        x_train, x_val, y_train, y_val = self.test_train_split()
        plot_imgs(org_imgs=x_val, mask_imgs=y_val, pred_imgs=a_binary, nm_img_to_plot=10)
