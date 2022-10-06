# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 21:13:14 2022

@author: DinhVu
"""

import argparse
import cv2
import os
from yolov5facedetector.face_detector import YoloDetector

def crop_pred(img,predict_value):
    bboxes, confs, points = predict_value
    yct_E = (points[0][0][0][1] + points[0][0][1][1])/2
    yct_EN = (yct_E + points[0][0][2][1] )* (0.5)
    # print(bboxes[0][0][1],int(yct_EN),bboxes[0][0][0],bboxes[0][0][2])
    img_array2 = img[bboxes[0][0][1]:int(yct_EN),bboxes[0][0][0]:bboxes[0][0][2]]
    return img_array2

def crop_eyes(img,predict_value):
    bboxes, confs, points = predict_value
    
    yct_EN = ((points[0][0][0][1] + points[0][0][1][1])/2 + points[0][0][2][1] )* (0.5)
    xct_EN = (((points[0][0][0][0] + points[0][0][1][0])/2) + points[0][0][2][0])/2
    x1 = points[0][0][0][0] * 2 -xct_EN 
    y1 = points[0][0][0][1] * 2 -yct_EN 
    
    left_eyes = img[int(y1):int(yct_EN),int(x1):int(xct_EN)]
    
    # x2_0,y2_0 = (xct_EN,y1)
    # x2_1 = points[0][0][1][0] * 2 - x2_0 
    # y2_1 = points[0][0][1][1] * 2 - y2_0 
    x2_1 = points[0][0][1][0] * 2 - xct_EN 
    y2_1 = points[0][0][1][1] * 2 - yct_EN 
    # print(y2_1,yct_EN,xct_EN,xct_EN)
    right_eyes=img[int(y2_1):int(yct_EN),int(xct_EN):int(x2_1)]
    
    return left_eyes,right_eyes
def crop_face(img,predict_value):
    bboxes, confs, points = predict_value
    img2 = img[bboxes[0][0][1]:bboxes[0][0][3],bboxes[0][0][0]:bboxes[0][0][2]]
    return img2

def add_name(img,predict_value,name):
    bboxes, confs, points = predict_value

    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    tf = max(tl - 1, 2)
    cv2.rectangle(img, (bboxes[0][0][0],bboxes[0][0][1]), (bboxes[0][0][2], bboxes[0][0][3]), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)
    cv2.putText(img, name, (bboxes[0][0][0], bboxes[0][0][1] - 2), 0, tl / 3, (255, 0, 0), thickness=tf, lineType=cv2.LINE_AA)
    return img
    
def add_eyes_stable(img,predict_value,left_eyes_stable,right_eyes_stable):
    """ """
    bboxes, confs, points = predict_value
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    tf = max(tl - 1, 1)
    green = (0,255,0)
    red = (225,0,0)

    if left_eyes_stable:
        color_left_eyes = green
    else:
        color_left_eyes = red
        
    if right_eyes_stable:
        color_right_eyes = green
    else:
        color_right_eyes = red
    
    
    #left_eyes
    
    yct_EN = ((points[0][0][0][1] + points[0][0][1][1])/2 + points[0][0][2][1] )* (0.5)
    xct_EN = (((points[0][0][0][0] + points[0][0][1][0])/2) + points[0][0][2][0])/2
    x1 = points[0][0][0][0] * 2 -xct_EN 
    y1 = points[0][0][0][1] * 2 -yct_EN 
    
    cv2.rectangle(img, (int(x1),int(y1)), (int(xct_EN)-1, int(yct_EN)), color_left_eyes, thickness=tl, lineType=cv2.LINE_AA)
    cv2.putText(img, str(left_eyes_stable),(int(x1),int(y1)-2), 0, tl / 3, color_left_eyes, thickness=tf, lineType=cv2.LINE_AA)
    
    # right_eyes
    

    x2_1 = points[0][0][1][0] * 2 - xct_EN 
    y2_1 = points[0][0][1][1] * 2 - yct_EN 
    cv2.rectangle(img, (int(xct_EN)+1,int(yct_EN)), (int(x2_1), int(y2_1)), color_right_eyes, thickness=tl, lineType=cv2.LINE_AA)
    cv2.putText(img, str(right_eyes_stable),(int(xct_EN),int(y2_1)-2), 0, tl / 3, color_right_eyes, thickness=tf, lineType=cv2.LINE_AA)
    return img
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', nargs='+', type=str, default='data/WIN_20220917_15_19_02_Pro.jpg', help='nơi lưu ảnh chưa cắt')
    parser.add_argument('--savepath', type=str, default='/data/result.jpg', help='nơi lưu hình ảnh sau cắt')  # file/folder, 0 for webcam
    parser.add_argument('--model_type', type=str, default='yolov5n ', help='model_type')  # file/folder, 0 for webcam

    opt = parser.parse_args()
    model = YoloDetector(target_size=1200,gpu=0,min_face=90,yolo_type='yolov5n')
    save_path = os.path.join(os.path.expanduser('~'),opt.savepath)
    crop_pred(model,opt.path,save_path)
    

    

 
# read image