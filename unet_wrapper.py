import numpy as np
import glob
import os
from skimage import io
import random

OUTPUT_IMG = "/Users/jiehyun/Jenna/UMassBoston/2022_Fall/CS696/01/output/test_img"
OUTPUT_MSK = "/Users/jiehyun/Jenna/UMassBoston/2022_Fall/CS696/01/output/test_mask"

class unet_wrapper:
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
    
    def convert_to_npy(self):
        #ref: https://gist.github.com/anilsathyan7/ffb35601483ac46bd72790fde55f5c04
        
        #Convert image to npy
        img_dirs = os.listdir(self.image_dir)
        img_dirs.sort()
        data_length = len(img_dirs)
        i = random.randint(1, data_length + 1)
        lists = []
        for item in img_dirs:   
            if os.path.isfile(self.image_dir + item):
                im = io.imread(self.image_dir + item)
                im = np.array(im)
                lists.append(im)
        imgset=np.array(lists)
        
        for i in range(data_length):
            if f'{i}.npy' not in OUTPUT_IMG:
                np.save(OUTPUT_IMG + f"/{i}.npy", imgset[i])

        #convert mask to npy
        msk_dirs = os.listdir(self.mask_dir)
        msk_dirs.sort()
        data_length_m = len(msk_dirs)
        mlists = []
        for item in msk_dirs:
            if os.path.isfile(self.mask_dir + item):
                msk = io.imread(self.mask_dir + item)
                msk = np.array(msk)
                mlists.append(msk)
        mskset=np.array(mlists)

        for i in range(data_length_m):
            if f'{i}.npy' not in OUTPUT_MSK:
                np.save(OUTPUT_MSK + f"/{i}.npy", mskset[i])

    def load_npy(self):
        images = sorted(glob.glob(self.image_dir + '/*.npy'))
        masks = sorted(glob.glob(self.mask_dir + '/*.npy'))

        imgs_list = []
        masks_list = []
        for image, mask in zip(images, masks):
            imgs_list.append(np.array(np.load(image, allow_pickle=True)))
            masks_list.append(np.array(np.load(mask, allow_pickle=True)))

        imgs_np = np.asarray(imgs_list)
        masks_np = np.asarray(masks_list)

        #print(imgs_np.shape, masks_np.shape)

        from keras_unet.utils import plot_imgs
        plot_imgs(org_imgs=imgs_np, mask_imgs=masks_np, nm_img_to_plot=3, figsize=5)

        x = np.asarray(imgs_np,dtype=np.float32)/255
        y = np.asarray(masks_np,dtype=np.float32)/255
        y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)

        #print(x.shape, y.shape)

        return x, y

    def train(self, x, y):
        from sklearn.model_selection import train_test_split
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.145, random_state=0)

        from keras_unet.utils import get_augmented
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
        
        from keras_unet.models import custom_unet
        input_shape = x_train[0].shape

        model = custom_unet(
            input_shape,
            filters=32,
            use_batch_norm=True,
            dropout=0.3,
            dropout_change_per_layer=0.0,
            num_layers=5
        )

        from keras.callbacks import ModelCheckpoint

        model_filename = 'segm_model_v3.h5'
        callback_checkpoint = ModelCheckpoint(
            model_filename, 
            verbose=1, 
            monitor='val_loss', 
            save_best_only=True,
        )

        from keras.optimizers import Adam
        from keras_unet.metrics import iou, iou_thresholded, dice_coef

        model.compile(
            optimizer=Adam(), 
            loss='binary_crossentropy',
            metrics=[iou, iou_thresholded, dice_coef, 'accuracy', 'sparse_categorical_accuracy']
        )

        history = model.fit(
        train_gen,
        steps_per_epoch=100,
        epochs=50,
        
        validation_data=(x_val, y_val),
        callbacks=[callback_checkpoint]
        )

        return model, x_val, y_val


    def predict(self, model, x_val):
        model_filename = 'segm_model_v3.h5'

        model.load_weights(model_filename)
        y_pred = model.predict(x_val)

        a = y_pred
        a_binary = np.zeros(a.shape, dtype=np.bool)
        a_binary[a > 0.1] = True

        return a_binary

    def plot_resuts(self, a_binary, x_val, y_val):
        from keras_unet.utils import plot_imgs
        plot_imgs(org_imgs=x_val, mask_imgs=y_val, pred_imgs=a_binary, nm_img_to_plot=10)