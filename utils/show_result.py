# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 21:18:06 2022

@author: DinhVu
"""

import cv2


def Show_name(img, predict_value, name):
    bboxes, confs, points = predict_value

    h, w, c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    tf = max(tl - 1, 2)
    cv2.rectangle(img, (bboxes[0][0][0], bboxes[0][0][1]), (bboxes[0][0][2], bboxes[0][0][3]), (0, 255, 0),
                  thickness=tl, lineType=cv2.LINE_AA)
    cv2.putText(img, name, (bboxes[0][0][0], bboxes[0][0][1] - 2), 0, tl / 3, (255, 0, 0), thickness=tf,
                lineType=cv2.LINE_AA)
    return img


def Show_eyes_stable(img, predict_value, left_eyes_stable, right_eyes_stable):
    """ """
    bboxes, confs, points = predict_value
    h, w, c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    tf = max(tl - 1, 1)
    green = (0, 255, 0)
    red = (225, 0, 0)

    if left_eyes_stable:
        color_left_eyes = green
    else:
        color_left_eyes = red

    if right_eyes_stable:
        color_right_eyes = green
    else:
        color_right_eyes = red

    # left_eyes

    yct_EN = ((points[0][0][0][1] + points[0][0][1][1]) / 2 + points[0][0][2][1]) * (0.5)
    xct_EN = (((points[0][0][0][0] + points[0][0][1][0]) / 2) + points[0][0][2][0]) / 2
    if left_eyes_stable is not None:
        x1 = points[0][0][0][0] * 2 - xct_EN
        y1 = points[0][0][0][1] * 2 - yct_EN

        cv2.rectangle(img, (int(x1), int(y1)), (int(xct_EN) - 1, int(yct_EN)), color_left_eyes, thickness=tl,
                      lineType=cv2.LINE_AA)
        cv2.putText(img, str(left_eyes_stable), (int(x1), int(y1) - 2), 0, tl / 3, color_left_eyes, thickness=tf,
                    lineType=cv2.LINE_AA)

    # right_eyes
    if right_eyes_stable is not None:
        x2_1 = points[0][0][1][0] * 2 - xct_EN
        y2_1 = points[0][0][1][1] * 2 - yct_EN
        cv2.rectangle(img, (int(xct_EN) + 1, int(yct_EN)), (int(x2_1), int(y2_1)), color_right_eyes, thickness=tl,
                      lineType=cv2.LINE_AA)
        cv2.putText(img, str(right_eyes_stable), (int(xct_EN), int(y2_1) - 2), 0, tl / 3, color_right_eyes, thickness=tf,
                    lineType=cv2.LINE_AA)
    return img
