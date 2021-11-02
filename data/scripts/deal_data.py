# encoding: utf-8
"""
@author: red0orange
@file: deal_data.py
@time: 2021/6/11
@desc: 处理原本的样本数据格式得到我们的样本数据格式
"""
import os
import cv2
import imghdr
import shutil
import random

import numpy as np

from utils import *


def move_files():
    data_root = "/home/cell/hdh/yolov5/data/RM_data/DJI ROCO"
    image_save_root = "/home/cell/hdh/yolov5/data/RM_data/images"
    xml_save_root = "/home/cell/hdh/yolov5/data/RM_data/xmls"

    ##### 先整理并重命名所有图像至一个文件夹，所有标注至另一个对称文件夹，并检查、丢弃损坏的数据
    sub_dirs = ["robomaster_Central China Regional Competition", "robomaster_Final Tournament", "robomaster_North China Regional Competition"]
    valid_data_dict = {}
    all_image_cnt = 0
    unvalid_cnt = 0
    for sub_dir in sub_dirs:
        image_root = os.path.join(data_root, sub_dir, "image")
        xml_root = os.path.join(data_root, sub_dir, "image_annotation")

        image_names = os.listdir(image_root)
        xml_names = [i.rsplit(".", maxsplit=1)[0] for i in os.listdir(xml_root)]
        valid_image_names = [i for i in image_names if i.rsplit(".", maxsplit=1)[0] in xml_names]  # 求交集
        valid_xml_names = [i.replace(i.rsplit(".", maxsplit=1)[1], "xml") for i in valid_image_names]
        unvalid_cnt += (len(valid_image_names) - len(image_names))
        all_image_cnt += len(image_names)

        image_paths = [os.path.join(image_root, i) for i in valid_image_names]
        xml_paths = [os.path.join(xml_root, i) for i in valid_xml_names]
        for i, image_path in enumerate(image_paths):
            if imghdr.what(image_path) is not None:
                valid_data_dict[image_path] = xml_paths[i]
            else:
                unvalid_cnt += 1
    print("error image: {} / {}".format(unvalid_cnt, all_image_cnt))
    print("valid image cnt: {}".format(len(valid_data_dict)))
    # 重命名并迁移
    os.makedirs(image_save_root, exist_ok=True)
    os.makedirs(xml_save_root, exist_ok=True)
    for i, (image_path, xml_path) in enumerate(valid_data_dict.items()):
        new_image_name = "{}".format(i) + "." + os.path.basename(image_path).rsplit('.', maxsplit=1)[1]
        new_xml_name = "{}".format(i) + "." + "xml"
        new_image_save_path = os.path.join(image_save_root, new_image_name)
        new_xml_save_path = os.path.join(xml_save_root, new_xml_name)
        shutil.copy(image_path, new_image_save_path)
        shutil.copy(xml_path, new_xml_save_path)


def xml_to_txt():
    import xml.dom.minidom as minidom
    xml_save_root = "/home/cell/hdh/yolov5/data/RM_data/xmls"
    label_save_root = "/home/cell/hdh/yolov5/data/RM_data/labels"
    os.makedirs(label_save_root, exist_ok=True)
    class_info_csv_path = "/home/cell/hdh/yolov5/data/RM_data/class_info.csv"
    xml_paths = get_files(xml_save_root, extensions=['.xml'])

    def get_boxes(xml_path, class_dict):
        class_names = []
        boxes = []
        infomations = []
        with open(xml_path, 'r') as f:
            dom = minidom.parse(f)
            root = dom.documentElement
            image_size = root.getElementsByTagName('size')[0]
            image_width = int(image_size.getElementsByTagName('width')[0].firstChild.data)
            image_height = int(image_size.getElementsByTagName('height')[0].firstChild.data)
            objects = root.getElementsByTagName('object')
            for object_ in objects:
                name = object_.getElementsByTagName('name')[0].firstChild.data
                box = object_.getElementsByTagName('bndbox')[0]

                x1 = box.getElementsByTagName('xmin')[0].firstChild.data
                y1 = box.getElementsByTagName('ymin')[0].firstChild.data
                x2 = box.getElementsByTagName('xmax')[0].firstChild.data
                y2 = box.getElementsByTagName('ymax')[0].firstChild.data
                x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
                boxes.append([x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height])
                class_names.append(name)

                attribute_names = [i.nodeName for i in object_.childNodes if i.nodeType == 1]
                exclude_names = ["bndbox", "name"]
                infomations.append({k: object_.getElementsByTagName(k)[0].firstChild.data for k in attribute_names if k not in exclude_names})
            return boxes, class_names, infomations

    # # 统计数据集的装甲板、车辆类别情况
    # all_class_names = []
    # all_car_armor_id = []
    # for i, xml_path in enumerate(xml_paths):
    #     boxes, class_names, infomations = get_boxes(xml_path, None)
    #     for ii, class_name in enumerate(class_names):
    #         all_class_names.append(class_name)
    #         if class_name == "armor":
    #             all_car_armor_id.append(infomations[ii]["armor_class"])
    #
    # from collections import Counter
    # print(Counter(all_class_names))
    # print(Counter(all_car_armor_id))

    class_dict = DatasetInfo(class_info_csv_path).train_class_dict()
    class_dict = {v: k for k, v in class_dict.items()}
    all_cnt = 0
    zero_cnt = 0
    for i, xml_path in enumerate(xml_paths):
        txt_save_path = os.path.join(label_save_root, os.path.basename(xml_path).rsplit('.',maxsplit=1)[0] + ".txt")
        boxes, class_names, infomations = get_boxes(xml_path, None)
        save_data = []
        for ii in range(len(boxes)):
            box = boxes[ii]
            class_name = class_names[ii]
            infomation = infomations[ii]
            category_name = None
            if class_name == "car":
                category_name = "机器人"
            elif class_name == "armor":
                id = infomation["armor_class"]
                color = infomation["armor_color"]
                if color != "gray" and id != "none" and int(id) <= 5:
                    if color == "red":
                        category_name = "红" + str(id) + "装甲板"
                    elif color == "blue":
                        category_name = "蓝" + str(id) + "装甲板"
            if category_name is not None:
                x1, y1, x2, y2 = box
                x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)
                save_data.append([class_dict[category_name], x, y, w, h])

        np.savetxt(txt_save_path, np.array(save_data), fmt='%.04f')
    pass


def split_dataset():
    split_ratio = 0.12
    image_save_root = "/home/cell/hdh/yolov5/data/RM_data/images"
    label_save_root = "/home/cell/hdh/yolov5/data/RM_data/color_labels"
    train_csv_save_path = "/home/cell/hdh/yolov5/data/RM_data/train_color.csv"
    valid_csv_save_path = "/home/cell/hdh/yolov5/data/RM_data/valid_color.csv"

    image_paths = get_image_files(image_save_root)
    label_paths = [os.path.join(label_save_root, os.path.basename(i).rsplit('.', maxsplit=1)[0] + ".txt") for i in image_paths]

    data_list = [[i, j] for i, j in zip(image_paths, label_paths)]
    random.shuffle(data_list)
    data_len = len(data_list)

    train_data_list = data_list[int(data_len * split_ratio):]
    valid_data_list = data_list[:int(data_len * split_ratio)]

    write_csv(train_data_list, train_csv_save_path)
    write_csv(valid_data_list, valid_csv_save_path)
    pass


def label_car_color():
    import xml.dom.minidom as minidom
    image_save_root = "/home/cell/hdh/yolov5/data/RM_data/images"
    xml_save_root = "/home/cell/hdh/yolov5/data/RM_data/xmls"
    label_save_root = "/home/cell/hdh/yolov5/data/RM_data/color_labels"
    class_info_csv_path = "/home/cell/hdh/yolov5/data/RM_data/class_info_color.csv"
    roi_car_image_save_paths = "/home/cell/hdh/yolov5/data/RM_data/car_images"
    label_color_root = "/home/cell/hdh/yolov5/data/RM_data/label_car_images"
    os.makedirs(roi_car_image_save_paths, exist_ok=True)

    image_paths = get_image_files(image_save_root)
    image_names = [os.path.basename(i) for i in image_paths]
    xml_paths = [os.path.join(xml_save_root, i.rsplit(".", maxsplit=1)[0] + ".xml") for i in image_names]
    label_paths = [os.path.join(label_save_root, i.rsplit(".", maxsplit=1)[0] + ".txt") for i in image_names]

    def get_boxes(xml_path, class_dict):
        class_names = []
        boxes = []
        infomations = []
        with open(xml_path, 'r') as f:
            dom = minidom.parse(f)
            root = dom.documentElement
            image_size = root.getElementsByTagName('size')[0]
            image_width = int(image_size.getElementsByTagName('width')[0].firstChild.data)
            image_height = int(image_size.getElementsByTagName('height')[0].firstChild.data)
            objects = root.getElementsByTagName('object')
            for object_ in objects:
                name = object_.getElementsByTagName('name')[0].firstChild.data
                box = object_.getElementsByTagName('bndbox')[0]

                x1 = box.getElementsByTagName('xmin')[0].firstChild.data
                y1 = box.getElementsByTagName('ymin')[0].firstChild.data
                x2 = box.getElementsByTagName('xmax')[0].firstChild.data
                y2 = box.getElementsByTagName('ymax')[0].firstChild.data
                x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
                boxes.append([x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height])
                class_names.append(name)

                attribute_names = [i.nodeName for i in object_.childNodes if i.nodeType == 1]
                exclude_names = ["bndbox", "name"]
                infomations.append({k: object_.getElementsByTagName(k)[0].firstChild.data for k in attribute_names if k not in exclude_names})
            return boxes, class_names, infomations

    class_dict = DatasetInfo(class_info_csv_path).train_class_dict()
    class_dict = {v: k for k, v in class_dict.items()}
    all_cnt = 0
    zero_cnt = 0
    for i, xml_path in enumerate(xml_paths):
        txt_save_path = os.path.join(label_save_root, os.path.basename(xml_path).rsplit('.',maxsplit=1)[0] + ".txt")
        boxes, class_names, infomations = get_boxes(xml_path, None)
        save_data = []
        for ii in range(len(boxes)):
            box = boxes[ii]
            class_name = class_names[ii]
            infomation = infomations[ii]
            category_name = None
            if class_name == "car":
                category_name = "机器人"
            elif class_name == "armor":
                id = infomation["armor_class"]
                color = infomation["armor_color"]
                if color != "gray" and id != "none" and int(id) <= 5:
                    if color == "red":
                        category_name = "红" + str(id) + "装甲板"
                    elif color == "blue":
                        category_name = "蓝" + str(id) + "装甲板"
            if category_name is not None:
                x1, y1, x2, y2 = box
                x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)
                if not(x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1):
                    save_data.append([category_name, x, y, w, h])

        # 处理得到车的颜色
        class_id_list = []
        for index, per_box in enumerate(save_data):
            print("index: {}".format(index))
            category_name, x, y, w, h = per_box
            class_id = None
            if category_name == "机器人":
                inside_boxes = []
                for per_other_box in save_data:
                    other_class_name, x1, y1, w1, h1 = per_other_box
                    if "机器人" not in other_class_name:
                        if abs(x1 - x) < (w / 2) and abs(y1 - y) < (h / 2):
                            inside_boxes.append(per_other_box)
                all_cnt += 1
                if len(inside_boxes) != 0:
                    class_id = class_dict["红机器人"] if 0 <= class_dict[inside_boxes[0][0]] <= 4 else class_dict["蓝机器人"]
                    pass
                else:
                    # # 保存进行标注
                    # image_path = image_paths[i]
                    # image_name = os.path.basename(image_path).rsplit('.', maxsplit=1)[0]
                    # roi_save_path = os.path.join(roi_car_image_save_paths, "{}_{}.jpg".format(image_name, index))
                    #
                    # image = cv2.imread(image_path)
                    # height, width = image.shape[:2]
                    # x *= width
                    # w *= width
                    # y *= height
                    # h *= height
                    # lx, rx = max(0, int(x - w / 2)), min(int(x + w / 2), width)
                    # uy, dy = max(0, int(y - h / 2)), min(int(y + h / 2), height)
                    # roi = image[uy:dy, lx:rx]
                    #
                    # cv2.imwrite(roi_save_path, roi)

                    image_path = image_paths[i]
                    image_name = os.path.basename(image_path).rsplit('.', maxsplit=1)[0]
                    if_blue_exist = os.path.exists(os.path.join(label_color_root, "blue", "{}_{}.jpg".format(image_name, index)))
                    if_red_exist = os.path.exists(os.path.join(label_color_root, "red", "{}_{}.jpg".format(image_name, index)))
                    if if_red_exist:
                        class_id = class_dict["红机器人"]
                    elif if_blue_exist:
                        class_id = class_dict["蓝机器人"]
                    else:
                        class_id = class_dict[random.choice(["红机器人", "蓝机器人"])]
                        print("warning: no label")
                    zero_cnt += 1
                class_id_list.append(class_id)
            else:
                class_id_list.append(class_dict[category_name])
                pass
        save_data = [[i, *j[1:]] for i, j in zip(class_id_list, save_data)]

        np.savetxt(txt_save_path, np.array(save_data), fmt='%.04f')
    print(all_cnt)
    print(zero_cnt)
    pass


if __name__ == '__main__':
    # move_files()
    # xml_to_txt()
    split_dataset()
    # label_car_color()
    pass
