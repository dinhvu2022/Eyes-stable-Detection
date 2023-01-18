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



def Triple_dataset( list_a, list_p, list_n, batch_size = 32):
    if len(list_a) != len(list_p) or len(list_a) != len(list_n):
        print('lenght achor, positive, negative is equal.')
        return None
    else:
        anchor_dataset = tf.data.Dataset.from_tensor_slices(list_a)
        positive_dataset = tf.data.Dataset.from_tensor_slices(list_p)
        negative_dataset = tf.data.Dataset.from_tensor_slices(list_n)

        dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
        dataset = dataset.shuffle(buffer_size=len(list_a))
        dataset = dataset.map(preprocess_triplets)
        train_dataset = dataset.batch(batch_size, drop_remainder=False)
        train_dataset = train_dataset.prefetch(8)
        return train_dataset





def Read_excel_apn(path):
    df = pd.read_excel(path,index_col=0)
    list_a = df['anchor']
    list_p = df['positive']
    list_n = df['negative']
    return list_a, list_p, list_n

def Create_apn_list(path_dir, save_folder = 'dataset/', incluce_train_test = False):
    if incluce_train_test:
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
        if save_folder:
            df = pd.DataFrame()
            df['anchor'] = a
            df['positive'] = p
            df['negative'] = n
            df.to_excel(save_folder + 'triplet.xlsx')
        return a, p, n
    else:
        if len(os.listdir(path_dir)) != 2:
            print('can not define train, test set')
            return None
        elif len(os.listdir(path_dir)[0]) > len(os.listdir(path_dir)[1]):
            train =  os.listdir(path_dir)[0]
            test = os.listdir(path_dir)[1]
        else:
            test =  os.listdir(path_dir)[0]
            train = os.listdir(path_dir)[1]
            
        a = []
        p = []
        n = []
        path_train = path_dir + train
        list_sub = sorted(os.listdir(path_train))
        for sub in list_sub:
            list_n_sub = os.listdir(path_train)
            list_n_sub.remove(sub)
            for img in os.listdir(path_train + sub):
                p_img = sorted(os.listdir(path_train+ sub))[0]
                for non_sub in list_n_sub:
                    n_img = sorted(os.listdir(path_train + non_sub))[0]
                    # n_img = random.choice(os.listdir('/content/all5c/' + non_sub))
                    a.append(path_train + sub + '/' + img)
                    p.append(path_train + sub + '/' + p_img)
                    n.append(path_train + non_sub + '/' + n_img )
        if save_folder:
            df = pd.DataFrame()
            df['anchor'] = a
            df['positive'] = p
            df['negative'] = n
            df.to_excel(save_folder + 'train.xlsx')
        
        a_t = []
        p_t = []
        n_t = []
        path_test = path_dir + test
        list_sub = sorted(os.listdir(path_test))
        for sub in list_sub:
            list_n_sub = os.listdir(path_test)
            list_n_sub.remove(sub)
            for img in os.listdir(path_test + sub):
                p_img = sorted(os.listdir(path_test+ sub))[0]
                for non_sub in list_n_sub:
                    n_img = sorted(os.listdir(path_test + non_sub))[0]
                    # n_img = random.choice(os.listdir('/content/all5c/' + non_sub))
                    a_t.append(path_test + sub + '/' + img)
                    p_t.append(path_test + sub + '/' + p_img)
                    n_t.append(path_test + non_sub + '/' + n_img )
        if save_folder:
            df = pd.DataFrame()
            df['anchor_test'] = a_t
            df['positive_test'] = p_t
            df['negative_test'] = n_t
            df.to_excel(save_folder + 'test.xlsx')

        return [a, p, n],[a_t, p_t, n_t]

                

def visualize(anchor, positive, negative):
    """Visualize a few triplets from the supplied batches."""

    def show(ax, image):
        # ax.imshow(image)
        ax.imshow(image[:,:,0], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=(3, 3))

    axs = fig.subplots(3, 3)
    for i in range(3):
        show(axs[i, 0], anchor[i])
        show(axs[i, 1], positive[i])
        show(axs[i, 2], negative[i])

