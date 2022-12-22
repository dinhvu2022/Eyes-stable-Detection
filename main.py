# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 21:13:58 2022

@author: DinhVu
"""

# import os
import winsound
import argparse
import cv2
import numpy as np
import tensorflow as tf

from yolov5facedetector.face_detector import YoloDetector

from utils.crop import Crop_eyes, Crop_face, Crop_pred
from utils.show_result import Show_name, Show_eyes_stable

from model.distance_model import DistanceLayer2


def Eyes_stable_model(weights_path='weights/Eyes_stable_model_best_07-0.04.hdf5'):
    model = tf.keras.applications.densenet.DenseNet169(weights=None, input_shape=(86, 86, 1), classes=2)
    model.load_weights(weights_path)
    return model


def Face_regco_model(weights_path='weights/Embedding_DenseNet.hdf5'):
    base_cnn = tf.keras.applications.densenet.DenseNet169(
        weights=None, input_shape=(224, 224, 1), include_top=False
    )

    flatten = tf.keras.layers.Flatten()(base_cnn.output)
    dense1 = tf.keras.layers.Dense(512, activation="relu")(flatten)
    dense1 = tf.keras.layers.BatchNormalization()(dense1)
    dense2 = tf.keras.layers.Dense(256, activation="relu")(dense1)
    dense2 = tf.keras.layers.BatchNormalization()(dense2)
    output = tf.keras.layers.Dense(256)(dense2)

    embedding = tf.keras.Model(base_cnn.input, output, name="Embedding")
    embedding.load_weights(weights_path)
    return embedding


def Face_regco_model_2(weights_path='weights/dennet_mini.hdf5'):
    base_cnn = tf.keras.applications.densenet.DenseNet121(
        weights=None, input_shape=(224, 224, 1), include_top=False
    )

    flatten = tf.keras.layers.Flatten()(base_cnn.output)
    dense1 = tf.keras.layers.Dense(512, activation="relu")(flatten)
    dense1 = tf.keras.layers.BatchNormalization()(dense1)
    dense2 = tf.keras.layers.Dense(256, activation="relu")(dense1)
    dense2 = tf.keras.layers.BatchNormalization()(dense2)
    output = tf.keras.layers.Dense(256)(dense2)

    embedding = tf.keras.Model(base_cnn.input, output, name="Embedding")
    embedding.load_weights(weights_path)
    return embedding


def Load_tensor_test_file(file_path):
    ar = np.loadtxt(file_path, delimiter=',')
    tfar = tf.convert_to_tensor(ar, dtype=None, dtype_hint=None, name=None)
    return tfar


def process_img(img, img_size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, img_size)
    rgb_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

    rgb_tensor = tf.expand_dims(rgb_tensor, 0)
    return rgb_tensor


def process_img_2(img, img_size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, img_size)
    # rgb_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    image = tf.image.convert_image_dtype(img, tf.float32)
    image_out = tf.expand_dims(image, 0)
    # print(image_out)
    return image_out


def make_predict_tensor(predict_tensor, len_of_tensor=5):
    list_make_predict_tensor = []
    for i in range(len_of_tensor):
        list_make_predict_tensor.append(predict_tensor)
    aa = tf.concat(list_make_predict_tensor, 0)
    return aa


def load_tensor_test_file(file_path='data/test.out'):
    """Load file npy """
    ar = np.loadtxt(file_path, delimiter=',')
    tensor = tf.convert_to_tensor(ar, dtype=None, dtype_hint=None, name=None)
    return tensor


def distance_model(input1_name="dist_in1", input2_name="dist_in2", input_shape=256):
    dist_in1 = tf.keras.layers.Input(name=input1_name, shape=input_shape)
    dist_in2 = tf.keras.layers.Input(name=input2_name, shape=input_shape)
    distance = DistanceLayer2()(dist_in1,
                                dist_in2)
    model = tf.keras.Model(inputs=[dist_in1, dist_in2], outputs=distance)
    return model


def detect_face_name(emd_predict_face, test_face_tensor, list_name, model):
    out = model((emd_predict_face, test_face_tensor))
    index_max = tf.math.argmin(out).numpy()
    return str(index_max)
    # return list_name[index_max]


def eyes_stable_warning(left_eyes_stable1, left_eyes_stable2, right_eyes_stable1, right_eyes_stable2):
    if left_eyes_stable1 == 0 + left_eyes_stable2 + right_eyes_stable1 + right_eyes_stable2 == 0:
        winsound.Beep(2000, 1000)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/VID20221221100307.mp4')
    parser.add_argument('--path_npy_file', type=str, default='data/emd_21_12_2022_01-0.07-95.out')
    parser.add_argument('--weights_face_reg', type=str, default='weights/emd_21_12_2022_01-0.07.hdf5')
    parser.add_argument('--weights_eyes_stables', type=str, default='weights/Eyes_stable_model_best_07-0.04.hdf5')
    opt = parser.parse_args()

    Face_keypoint_model = YoloDetector(target_size=1200, gpu=1, min_face=1, yolo_type='yolov5n')
    Face_reg = Face_regco_model_2(opt.weights_face_reg)
    Eyes_stable_model = Eyes_stable_model(opt.weights_eyes_stables)
    Distance_model = distance_model('Emd', 'Test', 256)
    List_name = ['Dương', 'Nhất', 'An', 'd', 'Vu']
    Test_Face_Tensor = Load_tensor_test_file(opt.path_npy_file)
    Face_reg_size = Face_reg.input.type_spec.shape[1:-1]
    Eyes_stable_size = Eyes_stable_model.input.type_spec.shape[1:-1]

    cap = cv2.VideoCapture(opt.source)
    last_l_eyes_stable, last_r_eyes_stable = 1, 1
    while True:
        ret, img = cap.read()
        if ret:
            Face_keypoint_value = Face_keypoint_model.predict(img)
            # print(Face_keypoint_value)
            if len(Face_keypoint_value[1][0]) > 0:
                Face_Crop = Crop_pred(img, Face_keypoint_value)
                Face_Crop = process_img_2(img=Face_Crop, img_size=Face_reg_size)
                Emd_Predict_Face = Face_reg.predict(Face_Crop)
                # print(Emd_Predict_Face)
                face_name = detect_face_name(Emd_Predict_Face, Test_Face_Tensor, List_name, Distance_model)

                left_eyes_img, right_eyes_img = Crop_eyes(img, Face_keypoint_value)
                if left_eyes_img is not None:
                    left_eyes_tensor = process_img(left_eyes_img, Eyes_stable_size)
                    left_eyes_stable = Eyes_stable_model.predict(left_eyes_tensor)
                    left_eyes = np.argmax(left_eyes_stable[0])
                else:
                    left_eyes = 0
                if right_eyes_img is not None:
                    right_eyes_tensor = process_img(right_eyes_img, Eyes_stable_size)
                    right_eyes_stable = Eyes_stable_model.predict(right_eyes_tensor)
                    right_eyes = np.argmax(right_eyes_stable[0])
                else:
                    right_eyes = 0

                # eyes_stable_warning(left_eyes, right_eyes, last_l_eyes_stable, last_r_eyes_stable)
                print(face_name, left_eyes, right_eyes)
                last_l_eyes_stable, last_r_eyes_stable = left_eyes, right_eyes
                img = Show_name(img, Face_keypoint_value, face_name)
                img = Show_eyes_stable(img, Face_keypoint_value, left_eyes,
                                       right_eyes)
            else:
                print('No one')
            cv2.imshow('frame', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    print(1)
