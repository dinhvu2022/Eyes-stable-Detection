# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 21:11:42 2022

@author: DinhVu
"""

import argparse
# import cv2
import os
from yolov5facedetector.face_detector import YoloDetector


def Crop_pred(img, predict_value):
    bboxes, confs, points = predict_value
    yct_E = (points[0][0][0][1] + points[0][0][1][1]) / 2
    yct_EN = (yct_E + points[0][0][2][1]) * 0.5
    # print(bboxes[0][0][1],int(yct_EN),bboxes[0][0][0],bboxes[0][0][2])
    img_array2 = img[bboxes[0][0][1]:int(yct_EN), bboxes[0][0][0]:bboxes[0][0][2]]
    return img_array2


def Crop_eyes(img, predict_value, ):
    bboxes, confs, points = predict_value

    yct_EN = int(((points[0][0][0][1] + points[0][0][1][1]) / 2 + points[0][0][2][1]) * 0.5)
    xct_EN = int((((points[0][0][0][0] + points[0][0][1][0]) / 2) + points[0][0][2][0]) / 2)
    x1 = int(points[0][0][0][0] * 2 - xct_EN)
    y1 = int(points[0][0][0][1] * 2 - yct_EN)
    if x1 == xct_EN:
        c = x1
        d = xct_EN + 2
    elif x1 > xct_EN:
        c = xct_EN
        d = x1
    else:
        d = xct_EN
        c = x1
    if y1 == yct_EN:
        a = y1
        b = yct_EN + 2
    elif y1 > yct_EN:
        a = yct_EN
        b = y1
    else:
        b = yct_EN
        a = y1
    left_eyes = img[a:b, c:d]
    # print(a, b, c, d)
    # if int(y1) >= int(yct_EN) or int(x1) >= int(xct_EN):
    #     left_eyes = None
    # else:
    #     left_eyes = img[int(y1):int(yct_EN), int(x1):int(xct_EN)]

    # x2_0,y2_0 = (xct_EN,y1)
    # x2_1 = points[0][0][1][0] * 2 - x2_0 
    # y2_1 = points[0][0][1][1] * 2 - y2_0 
    x2_1 = int(points[0][0][1][0] * 2 - xct_EN)
    y2_1 = int(points[0][0][1][1] * 2 - yct_EN)
    if x2_1 == xct_EN:
        g = x2_1
        h = xct_EN + 2
    elif x2_1 > xct_EN:
        g = xct_EN
        h = x2_1
    else:
        g = x2_1
        h = xct_EN

    if y2_1 == yct_EN:
        e = y2_1
        f = yct_EN + 2
    elif y2_1 > yct_EN:
        e = yct_EN
        f = y2_1
    else:
        e = y2_1
        f = yct_EN
    right_eyes = img[e:f, g:h]
    # print(e, f, g, h)
    # print(y2_1,yct_EN,xct_EN,xct_EN)
    # if img[int(y2_1) >= int(yct_EN) or int(xct_EN) >= int(x2_1)]:
    #     right_eyes = None
    # else:
    #     right_eyes = img[int(y2_1):int(yct_EN), int(xct_EN):int(x2_1)]
    return left_eyes, right_eyes


def Crop_face(img, predict_value):
    bboxes, confs, points = predict_value
    img2 = img[bboxes[0][0][1]:bboxes[0][0][3], bboxes[0][0][0]:bboxes[0][0][2]]
    return img2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', nargs='+', type=str, default='data/WIN_20220917_15_19_02_Pro.jpg',
                        help='nơi lưu ảnh chưa cắt')
    parser.add_argument('--savepath', type=str, default='/data/result.jpg',
                        help='nơi lưu hình ảnh sau cắt')  # file/folder, 0 for webcam
    parser.add_argument('--model_type', type=str, default='yolov5n ', help='model_type')  # file/folder, 0 for webcam

    opt = parser.parse_args()
    model = YoloDetector(target_size=1200, gpu=0, min_face=90, yolo_type='yolov5n')
    save_path = os.path.join(os.path.expanduser('~'), opt.savepath)
    Crop_pred(model, opt.path, save_path)

# read image
