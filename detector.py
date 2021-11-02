import os

import cv2
import numpy as np
import torch

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression
from utils.datasets import letterbox


class Detector(object):
    def __init__(self, weights_path, image_size=(640, 640), conf_thres=0.5, iou_thres=0.5, device="cpu"):
        # Load model
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.img_size = image_size
        self.model = attempt_load(weights_path, map_location=device)  # load FP32 model

        self.stride = int(self.model.stride.max())  # model stride
        new_height = check_img_size(self.img_size[0], s=self.stride)  # check img_size
        new_widht = check_img_size(self.img_size[1], s=self.stride)  # check img_size
        self.img_size = (new_height, new_height)
        if self.device != 'cpu':
            self.model.half()  # to FP16
        pass

    def detect(self, image):
        # preprocess image
        # input cv numpy image
        image = letterbox(image, self.img_size, auto=False, scaleup=True)[0]
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)

        image_tensor = torch.from_numpy(image).to(self.device)
        image_tensor = image_tensor.float()
        if self.device != "cpu":
            image_tensor = image_tensor.half()
        image_tensor /= 255.0
        if image_tensor.ndimension() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # begin detect
        output, train_output = self.model(image_tensor, augment=False)
        pred = non_max_suppression(output, self.conf_thres, self.iou_thres, labels=[], multi_label=True)[0]
        return pred


if __name__ == '__main__':
    weights_path = "/home/cell/hdh/yolov5/runs/train/exp5/weights/last.pt"
    test_image_path = "/home/cell/hdh/yolov5/data/custom_data/images/ALL/19-1754/175400002.bmp"

    detector = Detector(weights_path, conf_thres=0.5, iou_thres=0.5, device="cpu")
    pred = detector.detect(cv2.imread(test_image_path))
    print(pred)