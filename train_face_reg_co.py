# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 20:40:07 2022

@author: DinhVu
"""

import argparse
import tensorflow as tf

from model.Face_reg_model import DistanceLayer, SiameseModel
from dataset.Face_recg_dataset import Triple_dataset, Create_apn_list, Read_excel_apn, visualize


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
    parser.add_argument('--data', type=str, help = 'triple data',default='data')
    parser.add_argument('--tptrain', type=str, help = 'triple data',default='train.xlsx')
    parser.add_argument('--tptest', type=str, help = 'triple data',default='test.xlsx')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_last', type=bool, default=True)
    parser.add_argument('--only_save_best', type=bool, default=True)

    opt = parser.parse_args()

    if opt.tptrain == None and opt.test == None:
        "creat .xlsx"
        train_triplet, test_triplet = Create_apn_list(opt.data, incluce_train_test = True)
        trainset = Triple_dataset(train_triplet[0], train_triplet[1], train_triplet[2], batch_size = 16)
        testset = Triple_dataset(test_triplet[0], test_triplet[1], test_triplet[2], batch_size = 16)
    
    elif opt.tptrain[:-4] == '.xlsx' and opt.tptest[:-4] == '.xlsx':
        print('.xlsx')
        train_triplet = Read_excel_apn(opt.tptrain)
        test_triplet = Read_excel_apn(opt.tptest)
        trainset = Triple_dataset(train_triplet[0], train_triplet[1], train_triplet[2], batch_size = 16)
        testset = Triple_dataset(test_triplet[0], test_triplet[1], test_triplet[2], batch_size = 16)
    else:
        print('tptrain, tptrain is .xlsx or None')
    

    visualize(*list(trainset.take(1).as_numpy_iterator())[0])

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
        save_best_only=opt.only_save_best)
    history = siamese_model.fit(trainset, epochs=opt.epochs, validation_data=testset, callbacks=[model_checkpoint_callback])
