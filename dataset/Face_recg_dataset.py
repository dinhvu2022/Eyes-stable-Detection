# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 20:40:47 2022

@author: DinhVu
"""

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

def preprocess_image(filename, target_shape = (224, 224)):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """
    # print(filename)
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)

    # image = tf.image.random_brightness(image, 0.2, seed=None)
    return image


def preprocess_triplets(anchor, positive, negative):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """


    return (
        preprocess_image(anchor,),
        preprocess_image(positive,),
        preprocess_image(negative,),
    )

def Create_dataset(list_a,list_p,list_n):
    """tạo dataset cho tripless loss với 3 list anchor, 
        positive, nagative đã có trước"""
    image_count = len(list_a)
    print(image_count)
    anchor_dataset = tf.data.Dataset.from_tensor_slices(list_a)
    positive_dataset = tf.data.Dataset.from_tensor_slices(list_p)
    negative_dataset = tf.data.Dataset.from_tensor_slices(list_n)
    # negative_dataset = negative_dataset.shuffle(buffer_size=4096)
    
    dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
    dataset = dataset.shuffle(buffer_size=image_count)
    dataset = dataset.map(preprocess_triplets)
    
    # Let's now split our dataset in train and validation.
    train_dataset = dataset.take(round(image_count * 0.8))
    val_dataset = dataset.skip(round(image_count * 0.8))
    
    train_dataset = train_dataset.batch(64, drop_remainder=False)
    train_dataset = train_dataset.prefetch(8)
    
    val_dataset = val_dataset.batch(64, drop_remainder=False)
    val_dataset = val_dataset.prefetch(8)
    return train_dataset, val_dataset

def Read_excel_apn(path):
    df = pd.read_excel('/content/large_dataset.xlsx',index_col=0)
    list_a = df['anchor']
    list_p = df['positive']
    list_n = df['negative']
    return list_a, list_p, list_n

def Create_apn_list(path_dir, save_excel_path = 'dataset/large_dataset.xlsx'):
    a = []
    p = []
    n = []
    list_sub = sorted(os.listdir(path_dir))
    for sub in list_sub:
        list_n_sub = os.listdir(path_dir)
        list_n_sub.remove(sub)
        for img in os.listdir(path_dir + sub):
            p_img = sorted(os.listdir(path_dir+ sub))[0]
            for non_sub in list_n_sub:
                n_img = sorted(os.listdir(path_dir + non_sub))[0]
                # n_img = random.choice(os.listdir('/content/all5c/' + non_sub))
                a.append(path_dir + sub + '/' + img)
                p.append(path_dir + sub + '/' + p_img)
                n.append(path_dir + non_sub + '/' + n_img )
    
    if save_excel_path:
        df = pd.DataFrame()
        df['anchor'] = a
        df['positive'] = p
        df['negative'] = n
        df.to_excel(save_excel_path)
    return a, p, n

def visualize(anchor, positive, negative):
    """Visualize a few triplets from the supplied batches."""

    def show(ax, image):
        # ax.imshow(image)
        ax.imshow(image[:,:,0], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=(9, 9))

    axs = fig.subplots(7, 3)
    for i in range(7):
        show(axs[i, 0], anchor[i])
        show(axs[i, 1], positive[i])
        show(axs[i, 2], negative[i])

