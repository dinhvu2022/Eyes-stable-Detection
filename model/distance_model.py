# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 21:06:14 2022

@author: DinhVu
"""
import tensorflow as tf




class DistanceLayer2(tf.keras.layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)

        return ap_distance 


