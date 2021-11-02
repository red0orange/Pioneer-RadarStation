# encoding: utf-8
"""
@author: red0orange
@file: detect_video.py
@time: 2021/6/12
@desc:
"""
import os
import cv2
import torch
import time
import numpy as np
from detector import Detector
from utils.general import scale_coords, xyxy2xywh, xywh2xyxy, bbox_iou, box_iou


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


if __name__ == '__main__':
    video_path = "/home/cell/hdh/yolov5/data/RM_data/1080P.avi"
    model_checkout_path = "/home/cell/hdh/yolov5/runs/train/exp/weights/last.pt"
    save_video_path = "/home/cell/hdh/yolov5/data/RM_data/detect_1080P_7_17.avi"
    class_dict = {
        0: "red 1", 1: "red 2", 2: "red 3", 3: "red 4", 4: "red 5",
        5: "blue 1", 6: "blue 2", 7: "blue 3", 8: "blue 4", 9: "blue 5",
        10: "red car", 11: "blue car"
    }
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    detector = Detector(model_checkout_path, (1280, 1280), conf_thres=0.5, iou_thres=0.5, device="cuda")
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    save_video = cv2.VideoWriter(save_video_path, fourcc, 30.0, (1920, 1080), True)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("video finish !")
            break
        t1 = time_synchronized()
        result_save_frame = frame.copy()
        image_shape = frame.shape[:2]
        pred = detector.detect(frame)

        cur_image_shape = detector.img_size
        native_image_shape = image_shape[:2]
        scale_coords(cur_image_shape, pred[:, :4], native_image_shape)

        for i, box in enumerate(pred):
            x1, y1, x2, y2, conf, id = box
            name = class_dict[int(id)]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(result_save_frame, (x1, y1), (x2, y2), (0, 127, 255), 2)
            cv2.putText(result_save_frame, name, (x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0, 127, 255))
        save_video.write(result_save_frame)
        # if save_image is None:
        #     save_image = result_save_frame
        #     cv2.imwrite("test.png", save_image)
        #     break
        t2 = time_synchronized()
        print(f'per frame time: ({t2 - t1:.3f}s)')
    pass