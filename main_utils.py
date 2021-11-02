import os
import warnings
import csv
import shutil
import torch
import mimetypes
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import xml.dom.minidom as minidom
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QRect


def ifnone(a,b):
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a


def poxis2str(l):
    return [str(i) for i in l]


def _path_to_same_str(p_fn):
    "path -> str, but same on nt+posix, for alpha-sort only"
    s_fn = str(p_fn)
    s_fn = s_fn.replace('\\','.')
    s_fn = s_fn.replace('/','.')
    return s_fn


def _get_files(parent, p, f, extensions):
    p = Path(p)#.relative_to(parent)
    if isinstance(extensions,str): extensions = [extensions]
    low_extensions = [e.lower() for e in extensions] if extensions is not None else None
    res = [p/o for o in f if not o.startswith('.')
           and (extensions is None or f'.{o.split(".")[-1].lower()}' in low_extensions)]
    return res


def get_files(path, extensions=None, recurse:bool=False, exclude=None,
              include=None, presort:bool=False, followlinks:bool=False):
    "Return list of files in `path` that have a suffix in `extensions`; optionally `recurse`."
    if recurse:
        res = []
        for i,(p,d,f) in enumerate(os.walk(path, followlinks=followlinks)):
            # skip hidden dirs
            if include is not None and i==0:   d[:] = [o for o in d if o in include]
            elif exclude is not None and i==0: d[:] = [o for o in d if o not in exclude]
            else:                              d[:] = [o for o in d if not o.startswith('.')]
            res += _get_files(path, p, f, extensions)
        if presort: res = sorted(res, key=lambda p: _path_to_same_str(p), reverse=False)
        return poxis2str(res)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, path, f, extensions)
        if presort: res = sorted(res, key=lambda p: _path_to_same_str(p), reverse=False)
        return poxis2str(res)


def get_image_files(path, extensions=None, recurse:bool=False, exclude=None,include=None, presort:bool=False, followlinks:bool=False):
    image_extensions = set(k for k, v in mimetypes.types_map.items() if v.startswith('image/'))
    extensions = ifnone(extensions, image_extensions)
    return get_files(path, extensions, recurse, exclude, include, presort, followlinks)


def get_image_extensions():
    return list(set(k for k, v in mimetypes.types_map.items() if v.startswith('image/')))


class DatasetInfo(object):
    # 一个封装类别信息的类，传入类别信息的csv路径，即可有一些方便的函数，自行查看使用
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(path)
        pass

    def class_transform(self):
        class_transform = self.df[['origin_class_id', 'train_class_id']].to_numpy()
        class_transform = {k: v for k, v in class_transform}
        return class_transform

    def origin_class_dict(self):
        class_transform = self.df[['origin_class_id', 'origin_class_name']].to_numpy()
        class_transform = {k: v for k, v in class_transform}
        return class_transform

    def train_class_dict(self):
        class_transform = self.df[['train_class_id', 'train_class_name']].to_numpy()
        class_transform = {k: v for k, v in class_transform}
        return class_transform

    @staticmethod
    def create_dataset_info_csv_by_class_count(path, ori_class_id, ori_class_name, class_count, class_num_thres=100):
        assert len(ori_class_id) == len(ori_class_name) == len(class_count), '长度不相等不正常'
        train_class_id = []
        i = 1
        for class_id, class_num in zip(ori_class_id, class_count):
            if class_num <= class_num_thres:
                train_class_id.append(0)
            else:
                train_class_id.append(i)
                i += 1
        DatasetInfo.create_dataset_info_csv(path, ori_class_id, ori_class_name, train_class_id, class_count)
        pass

    @staticmethod
    def create_dataset_info_csv(path, ori_class_id, ori_class_name, train_class_id, class_count):
        assert len(ori_class_id) == len(ori_class_name) == len(train_class_id) == len(class_count), '长度不相等不正常'
        df = pd.DataFrame()
        df['origin_class_id'] = ori_class_id
        df['train_class_id'] = train_class_id
        df['ori_class_name'] = ori_class_name
        df['class_count'] = class_count
        df.to_csv(path, index=False)
        pass


def parse_xml_to_boxes(xml_path):
    "将标注结果xml解析为boxes矩阵"
    boxes = []
    names = []
    with open(xml_path, 'r', encoding='utf-8') as f:
        dom = minidom.parse(f)
        root = dom.documentElement
        objects = root.getElementsByTagName('object')
        for object_ in objects:
            categories_id = object_.getElementsByTagName('categories_id')[0].firstChild.data
            name = object_.getElementsByTagName('name')[0].firstChild.data
            x = object_.getElementsByTagName('x')[0].firstChild.data
            y = object_.getElementsByTagName('y')[0].firstChild.data
            width = object_.getElementsByTagName('width')[0].firstChild.data
            height = object_.getElementsByTagName('height')[0].firstChild.data
            categories_id = int(categories_id)
            x, y, width, height = map(float, [x, y, width, height])
            boxes.append([int(categories_id), x, y, width, height])
            names.append(name)
    return np.array(boxes), names


def parse_txt_to_boxes(txt_path):
    "return box xyxy"
    data = np.loadtxt(txt_path)
    if len(data.shape) == 1:
        if not data.shape[0]:
            return np.array([])
        data = data[None, ...]
    data[:, 1] -= (data[:, 3] / 2)
    data[:, 2] -= (data[:, 4] / 2)
    data[:, 3] += data[:, 1]
    data[:, 4] += data[:, 2]
    return data


def change_xml(xml_root, xml_paths, label_root):
    "将所有xml转为对应txt文件的脚本"
    for xml_path in xml_paths:
        rel_path = os.path.relpath(xml_path, xml_root)
        old_path = xml_path
        new_path = os.path.join(label_root, rel_path)
        new_path = new_path.rsplit('.', maxsplit=1)[0] + '.txt'
        if not os.path.isdir(os.path.dirname(new_path)):
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
        boxes, names = parse_xml_to_boxes(old_path)

        # 转换格式
        if len(boxes.shape) == 1:
            print('empty label: ', old_path)
            boxes = boxes[None, ...]
            with open(new_path, 'w') as txt:
                pass
            continue
        boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
        boxes[:, 2] = boxes[:, 2] + boxes[:, 4] / 2
        with open(new_path, 'w') as txt:
            for box_i in range(boxes.shape[0]):
                cat_id, x, y, width, height = boxes[box_i, :]
                txt.write('{} {:.3} {:.3} {:.3} {:.3}'.format(cat_id, x, y, width, height) + '\n')


def label_transform(ori_label_root, new_label_root, csv_path):
    "根据我们设定的csv文件内容，将标注结果中的类别进行脚本化处理"
    dsi = DatasetInfo(csv_path)
    class_transform = dsi.class_transform()
    old_label_files = get_files(ori_label_root, extensions=['.txt'], recurse=True)
    for i, label_file in enumerate(old_label_files):
        rel_path = os.path.relpath(label_file, ori_label_root)
        new_save_path = os.path.join(new_label_root, rel_path)
        boxes = np.loadtxt(label_file).reshape(-1, 5)
        for ii in range(len(boxes)):
            boxes[ii, 0] = class_transform[int(boxes[ii, 0])]
        boxes = boxes[boxes[:, 0] != -1]
        os.makedirs(os.path.dirname(new_save_path), exist_ok=True)
        np.savetxt(new_save_path, boxes, fmt='%.04f')
    pass


def create_fig(figsize=(8, 8), row=1, col=1):
    fig = plt.figure(figsize=figsize)
    fig.tight_layout()
    axes = fig.subplots(row, col)
    return fig, axes


def plot_bars(axe, x_labels, ys, bar_width=0.28, x_label_rotation=0):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    if not isinstance(ys[0], list):
        ys = [ys]
    x_labels = [str(i) for i in x_labels]

    bar_num = len(ys)

    x = np.arange(len(x_labels))  # the label locations
    width = bar_num * bar_width  # the width of the bars

    rects = []
    for i in range(bar_num):
        rects.append(axe.bar(x - width / 2 + bar_width * (i + 0.5), ys[i], width / bar_num))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    axe.set_ylabel('Scores')
    axe.set_xticks(x)
    axe.set_xticklabels(x_labels)
    axe.set_xticklabels(axe.get_xticklabels(), rotation=x_label_rotation)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            axe.annotate('{}'.format(height),
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom')

    for rect in rects:
        autolabel(rect)


def cut_scale_images(image_paths, xml_paths, root, ratio, class_dict):
    crop_save_root = root/"crop_images_larger_{}".format(ratio)

    error_count = 0
    count = 0
    for image_path, xml_path in zip(image_paths, xml_paths):
        boxes, _ = parse_xml_to_boxes(xml_path)
        #     print(boxes)
        image = QImage(image_path)
        image_width, image_height = image.size().width(), image.size().height()

        crop_images = []
        class_names = []
        for i, box in enumerate(boxes):
            class_id, x, y, width, height = box
            class_name = class_dict[class_id]
            x, width = map(lambda i: i*image_width, [x, width])
            y, height = map(lambda i: i*image_height, [y, height])
            width += width * ratio
            height += height * ratio
            x -= (width * ratio) / 2
            y -= (height * ratio) / 2
            x, y, width, height = map(int, [x, y, width, height])
            # 错误判定
            #         if x1 <= 0 or y1 <= 0 or x2 >= image_width or y2 >= image_height or x1 >= x2 or y1 >= y2:
            #             print(f'error_image:{image_path}   {i}  ({x1},{y1},{x2},{y2})')
            #             error_count += 1
            #             continue
            crop_images.append(image.copy(QRect(x, y, width, height)))
            class_names.append(class_name)

        for i, (crop_image, class_name) in enumerate(zip(crop_images, class_names)):
            os.makedirs(os.path.join(crop_save_root, class_name), exist_ok=True)
            save_path = os.path.join(crop_save_root, class_name, f'{os.path.basename(image_path)}_{i}.jpg')
            count += 1
            crop_image.save(save_path)
    print(count)
    # print(error_count)
    pass


def cut_one_scale_image(image_path, boxes, ratio, save_root):
    # boxes的格式是[[class_id, x, y, width, height], ...]，class_id在这里没有用，可以全部设零
    # 其中的x、y是中心坐标，不是左上坐标，可以自己根据需求更改
    image = QImage(image_path)
    image_width, image_height = image.size().width(), image.size().height()

    crop_images = []
    class_names = []
    for i, box in enumerate(boxes):
        class_id, x, y, width, height = box
        x, width = map(lambda i: i*image_width, [x, width])
        y, height = map(lambda i: i*image_height, [y, height])
        width += width * ratio
        height += height * ratio
        x -= (width * ratio) / 2
        y -= (height * ratio) / 2
        x, y, width, height = map(int, [x, y, width, height])
        crop_images.append(image.copy(QRect(x, y, width, height)))
    count = 0
    for i, crop_image in enumerate(crop_images):
        save_path = os.path.join(save_root, f'{os.path.basename(image_path)}_{i}.jpg')
        count += 1
        crop_image.save(save_path)
    pass


def read_csv(csv_path):
    result = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            result.append(line)
    return result


def write_csv(data, csv_path):
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        for line in data:
            writer.writerow(line)
    pass


