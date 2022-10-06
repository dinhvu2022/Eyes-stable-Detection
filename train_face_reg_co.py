# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 20:40:07 2022

@author: DinhVu
"""

import argparse
import tensorflow as tf

from model.Face_reg_model import DistanceLayer, SiameseModel
from dataset.Face_recg_dataset import Create_dataset, Create_apn_list, Read_excel_apn, visualize


def base_model(name):
    base_cnn = tf.keras.applications.densenet.DenseNet169(
        weights=None, input_shape=(224, 224, 1), include_top=False
    )

    flatten = tf.keras.layers.Flatten()(base_cnn.output)
    dense1 = tf.keras.layers.Dense(512, activation="relu")(flatten)
    dense1 = tf.keras.layers.BatchNormalization()(dense1)
    dense2 = tf.keras.layers.Dense(256, activation="relu")(dense1)
    dense2 = tf.keras.layers.BatchNormalization()(dense2)
    output = tf.keras.layers.Dense(256)(dense2)

    trainable_stable = False
    for layers in base_cnn.layers:
        if layers.name == "conv5_block1_out":
            trainable_stable = True
        layers.trainable = trainable_stable
    model = tf.keras.Model(base_cnn.input, output, name=name)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=10)

    opt = parser.parse_args()

    anchor, positive, negative = [], [], []
    if opt.data[:-4] != '.xlsx':
        print('đọc dataset từ file xlsx')
        anchor, positive, negative = Read_excel_apn(opt.data)
    elif opt.data[:1] != '/':
        "tạo dataset chưa có sẵn file ano"
        anchor, positive, negative = Create_apn_list(opt.data, opt.data + '/dataset.xlsx')

    train, val = Create_dataset(anchor, positive, negative)

    visualize(*list(train.take(1).as_numpy_iterator())[0])

    embedding = base_model('Embedding')

    anchor_input = tf.keras.layers.Input(name="anchor", shape=(224, 224, 1))
    positive_input = tf.keras.layers.Input(name="positive", shape=(224, 224, 1))
    negative_input = tf.keras.layers.Input(name="negative", shape=(224, 224, 1))

    distances = DistanceLayer()(
        embedding(anchor_input),
        embedding(positive_input),
        embedding(negative_input),
    )

    siamese_network = tf.keras.Model(
        inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )

    siamese_model = SiameseModel(siamese_network)
    siamese_model.compile(optimizer=tf.keras.optimizers.Adam(0.0001))
    checkpoint_filepath = 'weights/face_reg_{epoch:02d}-{val_loss:.2f}.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=False)
    siamese_model.fit(train, epochs=opt.epochs, validation_data=val, callbacks=[model_checkpoint_callback])
