# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 21:13:43 2022

@author: DinhVu
"""

# import os
import argparse
# import shutil
import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type = int, default=(86,86))

    parser.add_argument('--train_dir', type = str, default='data')
    parser.add_argument('--val_dir', type = str, default='data')
    parser.add_argument('--batch_size', type = int, default=32)
    parser.add_argument('--epochs', type = int, default=10)

    parser.add_argument('--save_weights', type = str, default='weights')
    opt = parser.parse_args()
       
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory= opt.train_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size= opt.batch_size,
        color_mode = 'grayscale',
        image_size=opt.input_size)
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        directory= opt.val_dir,
        labels='inferred',
        label_mode='categorical',
        color_mode = 'grayscale',
        batch_size= opt.batch_size,
        image_size= opt.input_size ) 
    
    model = tf.keras.applications.densenet.DenseNet169(weights=None, 
                                                       input_shape=(86, 86, 1), 
                                                       classes=2)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), 
                      loss='categorical_crossentropy',metrics=['accuracy'])
    checkpoint_filepath = opt.save_weights +  '/Eyes_stable_model_{epoch:02d}-{val_loss:.2f}.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=False)
    model.fit(train_ds,epochs = opt.epochs, 
              validation_data = validation_ds,
              callbacks=[model_checkpoint_callback])