# 1. read recoginfo.txt
# 2. reformat detect data to suit the val code process in val.py

import os
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from typing import List
import xml.etree.ElementTree as ET


class quanzhi_eval:
    def __init__(self, recoginfo_path: str, draw_box: bool = False) -> None:
        self.recoginfo_path = recoginfo_path
        self.draw_box = draw_box
        self.class_list = ["wire", "shoes", "trashcan", "carpet", "clothes",
                      "feces", "scale", "socks", "stains", "fanbase", "paper"]

    def main(self) -> None:
        # recoginfo.txt must beside detect img. Like
        # └── pic
        # 	├── IMG_19700101_0000030416_1280x720.jpg
        # 	├── IMG_19700101_0000030416_1280x720.yuv
        # 	├── IMG_19700101_0000030566_1280x720.jpg
        # 	├── IMG_19700101_0000030566_1280x720.yuv
        # 	└── recoginfo.txt
        # then recoginfo_path = './pic/'
        recoginfo = pd.read_csv(os.path.join(self.recoginfo_path, 'recoginfo.txt'), sep=' |:',
                                names=['filename', 'x', 'x_min_v', 'y', 'y_min_v',
                                       'w', 'w_v', 'h', 'h_v', 'l', 'l_v', 'd', 'd_v'])
        recoginfo['x_max_v'] = recoginfo['x_min_v'] + recoginfo['w_v']
        recoginfo['y_max_v'] = recoginfo['y_min_v'] + recoginfo['h_v']
        self.recoginfo_array = np.array(recoginfo[['filename', 'l_v', 'x_min_v', 'y_min_v', 'x_max_v', 'y_max_v']])
        return

    def get_xywhc_with_imgnames(self, img_names: Tuple[str]) -> np.ndarray:
        res = []
        recoginfo_array_row_num = self.recoginfo_array.shape[0]
        for img_name in img_names:
            for i in range(recoginfo_array_row_num):
                # Get detect res of imgs in this batch
                if self.recoginfo_array[i, 0] == Path(img_name).stem + '_1280x720' + Path(img_name).suffix:
                    res = np.append(res, list(self.recoginfo_array[i, 1:]))
        res = np.reshape(res, (-1, 5))
        res = np.concatenate((res[:, 1:5], np.array([[0.75] * res.shape[0]]), res[:, 0:1]), axis=1)
        return [torch.tensor(res)]

    def generate_ground_truth_txt(self, folder: str) -> None:
        all_train_val_img_filepaths = [pathlib_object for pathlib_object in Path(folder).rglob(
            "*") if pathlib_object.suffix in ['.xml']]
        for file in all_train_val_img_filepaths:
            f = open("amicro_files/test_quanzhi_AIboard/dataset_for_val/ground-truth/" + file.stem + "_1280x720.txt", "w")
            xml_infos = self.parse_xml(file)
            for xml_info in xml_infos:
                f.write(' '.join(xml_info) + '\n')
            f.close()

    def generate_predict_res_txt(self) -> None:
        for i in range(self.recoginfo_array.shape[0]):
            # There is only one detected object per image.
            predict_res_info = [
                self.class_list[self.recoginfo_array[i, 1]], str(0.75), str(self.recoginfo_array[i, 2]),
                str(self.recoginfo_array[i, 3]), str(self.recoginfo_array[i, 4]),
                str(self.recoginfo_array[i, 5])]
            f = open("amicro_files/test_quanzhi_AIboard/dataset_for_val/detection-results/" +
                     Path(self.recoginfo_array[i, 0]).stem + ".txt", "w")
            f.write(' '.join(predict_res_info) + '\n')
            f.close()

    def parse_xml(self, file) -> List:
        tree = ET.parse(file)
        root = tree.getroot()
        xml_info = []
        cls_with_imagenames = {}
        for obj in root.iter('size'):
            width = obj.find('width').text
            height = obj.find('height').text
        for obj in root.iter('object'):
            cls = obj.find('name').text
            xmin = round(int(obj.find('bndbox').find('xmin').text) / int(width) * 1280)
            ymin = round(int(obj.find('bndbox').find('ymin').text) / int(height) * 720)
            xmax = round(int(obj.find('bndbox').find('xmax').text) / int(width) * 1280)
            ymax = round(int(obj.find('bndbox').find('ymax').text) / int(height) * 720)
            xml_info.append([str(cls), str(xmin),
                             str(ymin), str(xmax), str(ymax)])
        return xml_info


class gddi_eval(quanzhi_eval):
    def __init__(self, recoginfo_path: str, draw_box: bool = False) -> None:
        self.recoginfo_path = recoginfo_path
        self.draw_box = draw_box
        self.class_list =["carpet", "clothes", "data_wire", "fanbase",
        "feces", "paper", "shoes", "stains", "trashcan"]

    def main(self) -> None:
        # recoginfo.txt must beside detect img. Like
        # └── pic
        # 	├── IMG_19700101_0000030416_1280x720.jpg
        # 	├── IMG_19700101_0000030416_1280x720.yuv
        # 	├── IMG_19700101_0000030566_1280x720.jpg
        # 	├── IMG_19700101_0000030566_1280x720.yuv
        # 	└── recoginfo.txt
        # then recoginfo_path = './pic/'
        recoginfo = pd.read_csv(os.path.join(self.recoginfo_path, 'recoginfo.txt'), sep=' ',
                                names=['filename', 'filename_v', 'cls', 'cls_v', 'cls_name',
                                'cls_name_v', 'prob', 'prob_v', 'bbox:', 'x_min_v', 'y_min_v', 'w_v', 'h_v'])

        recoginfo['x_min_v'] = recoginfo['x_min_v'].astype(int)
        recoginfo['y_min_v'] = recoginfo['y_min_v'].astype(int)
        recoginfo['w_v'] = recoginfo['w_v'].astype(int)
        recoginfo['h_v'] = recoginfo['h_v'].astype(int)
        recoginfo['x_max_v'] = recoginfo['x_min_v'] + recoginfo['w_v']
        recoginfo['y_max_v'] = recoginfo['y_min_v'] + recoginfo['h_v']
        self.recoginfo_array = np.array(
            recoginfo[['filename_v', 'cls_v', 'x_min_v', 'y_min_v', 'x_max_v', 'y_max_v', 'prob_v']])
        return

    def get_xywhc_with_imgnames(self, img_names: Tuple[str]) -> np.ndarray:
        res = []
        recoginfo_array_row_num = self.recoginfo_array.shape[0]
        for img_name in img_names:
            for i in range(recoginfo_array_row_num):
                # Get detect res of imgs in this batch
                if self.recoginfo_array[i, 0] == Path(img_name).stem + Path(img_name).suffix:
                    res = np.append(res, list(self.recoginfo_array[i, 1:]))
        res = np.reshape(res, (-1, 6))
        return [torch.tensor(res)]

    def generate_ground_truth_txt(self, folder: str) -> None:
        all_train_val_img_filepaths = [pathlib_object for pathlib_object in Path(folder).rglob(
            "*") if pathlib_object.suffix in ['.xml']]
        for file in all_train_val_img_filepaths:
            f = open("amicro_files/test_quanzhi_AIboard/dataset_for_val/ground-truth/" + file.stem + ".txt", "w")
            xml_infos = self.parse_xml(file)
            for xml_info in xml_infos:
                f.write(' '.join(xml_info) + '\n')
            f.close()

    def generate_predict_res_txt(self) -> None:
        for i in range(self.recoginfo_array.shape[0]):
            predict_res_info = [
                self.class_list[self.recoginfo_array[i, 1]], str(self.recoginfo_array[i, 6]), str(self.recoginfo_array[i, 2]),
                str(self.recoginfo_array[i, 3]), str(self.recoginfo_array[i, 4]),
                str(self.recoginfo_array[i, 5])]
            f = open("amicro_files/test_quanzhi_AIboard/dataset_for_val/detection-results/" +
                     Path(self.recoginfo_array[i, 0]).stem + ".txt", "a")
            f.write(' '.join(predict_res_info) + '\n')
            f.close()

    def parse_xml(self, file) -> List:
        tree = ET.parse(file)
        root = tree.getroot()
        xml_info = []
        cls_with_imagenames = {}
        for obj in root.iter('size'):
            width = obj.find('width').text
            height = obj.find('height').text
        for obj in root.iter('object'):
            cls = obj.find('name').text
            xmin = round(int(obj.find('bndbox').find('xmin').text))
            ymin = round(int(obj.find('bndbox').find('ymin').text))
            xmax = round(int(obj.find('bndbox').find('xmax').text))
            ymax = round(int(obj.find('bndbox').find('ymax').text))
            xml_info.append([str(cls), str(xmin),
                             str(ymin), str(xmax), str(ymax)])
        return xml_info



if __name__ == '__main__':
    # gddi
    recoginfo_path = 'amicro_files/test_quanzhi_AIboard/dataset_for_val/gddi_val/'
    gddi_eval_object = gddi_eval(recoginfo_path)
    gddi_eval_object.main()
    # quanzhi_eval_object.get_xywhc_with_imgnames('/home/yousixia/project/yolov5/amicro_files/test_quanzhi_AIboard/dataset_for_val/pic/IMG_19700101_0000030416_1280x720.jpg')
    gddi_eval_object.generate_ground_truth_txt(
        '/data/yousixia/images/val_for_gddi/val')
    gddi_eval_object.generate_predict_res_txt()

    # quanzhi
    if 0:
        recoginfo_path = 'amicro_files/test_quanzhi_AIboard/dataset_for_val/pic'
        quanzhi_eval_object = quanzhi_eval(recoginfo_path)
        quanzhi_eval_object.main()
        # quanzhi_eval_object.get_xywhc_with_imgnames('/home/yousixia/project/yolov5/amicro_files/test_quanzhi_AIboard/dataset_for_val/pic/IMG_19700101_0000030416_1280x720.jpg')
        quanzhi_eval_object.generate_ground_truth_txt(
            '/data/yousixia/images/shujutang_data/shujutang_for_quanzhi_val_adjust_xml_class')
        quanzhi_eval_object.generate_predict_res_txt()
