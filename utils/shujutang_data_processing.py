import os
from collections import Counter
from glob import glob
import random
import shutil
from pathlib import Path
import platform
import threading
from multiprocessing import Process
import time
from tqdm import tqdm

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from data_analysis import get_name_from_xml


def construct_amicro_img_dir(saved_path, labelme_path, has_hard_sample: bool):
    # 2.创建要求文件夹
    saved_path = Path(saved_path)
    if not os.path.exists(saved_path / "Annotations"):
        os.makedirs(saved_path / "Annotations")
    if not os.path.exists(saved_path / "JPEGImages/"):
        os.makedirs(saved_path / "JPEGImages/")
    if not os.path.exists(saved_path / "ImageSets/Main/"):
        os.makedirs(saved_path / "ImageSets/Main/")

    xml_files = []
    image_files = []
    # 3.复制xml文件到 VOC2007_amicro/Annotations/下
    # 3.复制xml对应的jpg到 VOC2007_amicro/JPEGImages/下
    for path in labelme_path:
        xml_files = path.glob("*.xml")
        image_files = list(path.glob("*.*[jpg,png]"))
        for xml_file in tqdm(xml_files, desc="xml"):
            shutil.copy(xml_file, saved_path / "Annotations/")
        xml_files = path.glob("*.xml")
        xml_stem_list = [xml.stem for xml in list(xml_files)]
        for image in tqdm(image_files, desc="jpg"):
            if not has_hard_sample:
                # check if pic has a xml correlate to it.
                if image.stem in xml_stem_list:
                    shutil.copy(image, saved_path / "JPEGImages/")
            else:
                shutil.copy(image, saved_path / "JPEGImages/")

    print("Finished copy image files to {}/Annotations/".format(saved_path))
    print("Finished copy image files to {}/JPEGImages/".format(saved_path))

    # 5.split files for txt
    txtsavepath = saved_path / "ImageSets/Main/"
    ftrainval = open((txtsavepath / 'trainval.txt'), 'w')
    ftest = open((txtsavepath / 'test.txt'), 'w')
    ftrain = open((txtsavepath / 'train.txt'), 'w')
    fval = open((txtsavepath / 'val.txt'), 'w')
    trainval_files = glob(os.path.join(saved_path, "Annotations/*.xml"))
    trainval_files = [i.split("/")[-1].split(".xml")[0] for i in trainval_files]
    total_files = os.listdir(saved_path / "JPEGImages")
    test_files = [x for x in total_files if x not in trainval_files]
    test_files = [i.split("/")[-1].split(".jpg")[0] for i in test_files]
    for file in trainval_files:
        ftrainval.write(file + "\n")
    # test
    for file in test_files:
        ftest.write(file + "\n")
    # split
    train_files, val_files = train_test_split(trainval_files, test_size=0.15, random_state=42)
    # train
    for file in train_files:
        ftrain.write(file + "\n")

    # val
    for file in val_files:
        fval.write(file + "\n")

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()


if __name__ == '__main__':
    # --data-root /home/yousixia/data/images --images-folder-name indoor_detecting --target-folder-path /home/yousixia/data/VOCs/indoor_detecting
    parser = argparse.ArgumentParser(
        description='This program create '
                    'the directory structure like VOC and used to train yolo model')
    parser.add_argument('--data-root', "-r", type=str, default='/home/yousixia/data/images',
                        help='Directory for saving frame images')
    parser.add_argument('--images-folder-name', "-n", type=str,
                        help='Directory names including images and annatations, '
                             'Multiple folder names, separated by commas。（e.q. labelme_20210520,labelme）')
    parser.add_argument('--target-folder-path', "-p", type=str,
                        help='Specifying forder names saving processed training data, '
                             'it should locates under "/home/yousixia/data/VOCs"（e.q. /home/yousixia/data/VOCs/VOC2007_amicro）')
    parser.add_argument('--has-hard-sample', "-hard", default=True, action='store_false',
                        help="Include the images in folder which don't have correlated xml files")
    parser.add_argument('--test', "-t", default=True, action='store_false',
                        help="Test this script, no data destruction.")
    parser.add_argument('--test_directory', "-td", type=str,
                        help='Specifying forder names saving test data, will copy part of data to test this script. If \'--test\' is false, this won\'t work.')
    # parser.add_argument('--label-path', nargs='+', type=str, help='Directory containing xml labels')
    opt = parser.parse_args()
    opt.data_root = Path(opt.data_root).as_posix()

    if os.path.isdir(opt.target_folder_path):
        raise Exception('target-folder-path: "{}" existed, please remove it manually first.'.format(opt.target_folder_path))

    base_path = Path(opt.data_root)
    all_files = [pathlib_object for pathlib_object in base_path.rglob("*") if pathlib_object.suffix in ['.jpg', '.png', '.xml']]
    all_xml_files = [x for x in all_files if x.suffix == '.xml']
    all_img_files = [x for x in all_files if (x.suffix in ['.png', '.jpg']) and (
        '门槛' not in x.stem and '门-门框' not in x.stem and '门' not in x.stem)]

    all_xml_stem_files = [x.stem for x in all_xml_files]
    all_img_stem_files = [x.stem for x in all_img_files]
    all_xml_str_files = [str(x) for x in all_xml_files]
    all_img_str_files = [str(x) for x in all_img_files]
    # all_img_stem_files = set(all_img_stem_files)


    # 提取重复文件到txt中，反馈给数据堂
    # def write_rep_filepath_to_txt(all_stem_files, all_str_files, output_filename):
    #     f = open(output_filename + ".txt", "w")
    #     save_dict = {}
    #     rep_str_filepath = []
    #     for i, x in enumerate(tqdm(all_stem_files)):
    #         if x not in save_dict:
    #             save_dict[x] = 1
    #         elif save_dict[x] == 1:
    #             save_dict[x] += 1
    #         elif save_dict[x] > 1:
    #             f.write(all_str_files[i] + '\n')
    #             rep_str_filepath.append(all_str_files[i])
    #     f.close()
    # img_p = Process(target=write_rep_filepath_to_txt, args=(
    #     all_img_stem_files, all_img_str_files, "img_rep_filepath", ))
    # xml_p = Process(target=write_rep_filepath_to_txt, args=(
    #     all_xml_stem_files, all_xml_str_files, "xml_rep_filepath", ))
    # img_p.start()
    # xml_p.start()
    # img_p.join()
    # xml_p.join()

    # 移动遗漏标注的图片到另一个文件夹
    # not_labeled_stem = list(set(all_xml_stem_files) ^ set(all_img_stem_files))
    # all_not_labeled_img_files = [x for x in all_img_files if x.stem in not_labeled_stem]
    # for x in all_not_labeled_img_files:
    #     shutil.move(str(x), "/data/yousixia/images/shujutang_data/购买数据/遗漏标注图片")
    # 复制所有已标注xml和jpg到另一个文件夹下，以便复用构造VOC目录结构的脚本
    # for x in tqdm(all_img_str_files):
    #     shutil.copy(str(x), "/data/yousixia/images/shujutang_data/shujutang_to_VOC")
    # for x in tqdm(all_xml_str_files):
    #     shutil.copy(str(x), "/data/yousixia/images/shujutang_data/shujutang_to_VOC")

    # 核验是否有遗漏的xml和jpg文件
    shujutang_to_VOC_base_path = Path("/data/yousixia/images/shujutang_data/shujutang_to_VOC")
    # shujutang_to_VOC_all_files = [pathlib_object for pathlib_object in shujutang_to_VOC_base_path.rglob(
    #     "*") if pathlib_object.suffix in ['.jpg', '.xml']]
    # all_xml_files_str = [x.stem + x.suffix for x in all_xml_files]
    # all_img_files_str = [x.stem + x.suffix for x in all_img_files]
    # all_files_str = all_xml_files_str + all_img_files_str
    # shujutang_to_VOC_all_files_stem = [x.stem + x.suffix for x in shujutang_to_VOC_all_files]
    # print(list(set(all_files_str) ^ set(shujutang_to_VOC_all_files_stem)))

    # 抽取0.5%的图片用于val
    all_train_val_img_filepaths = [str(pathlib_object) for pathlib_object in shujutang_to_VOC_base_path.rglob(
        "*") if pathlib_object.suffix in ['.jpg']]

    if not os.path.exists("shujutang_VOC_img_filepath.txt"):
        f = open("shujutang_VOC_img_filepath.txt", "w")
        for i, x in enumerate(tqdm(all_train_val_img_filepaths)):
            f.write(x + '\n')
        f.close()

    dataset = pd.read_csv("shujutang_VOC_img_filepath.txt", header=None, encoding='utf-8')

    # split by the last '/'
    dataset = pd.concat([dataset, dataset[0].str.rsplit("/", n=1, expand=True)[1]], axis=1)
    dataset = pd.concat([dataset, dataset[1].str.split('\_|\.', expand=True)], axis=1)
    dataset.columns = ["path", "filename", "class", "house_no", "ID", "suffix"]
    # dataset = dataset.sort_values(['class', 'house_no'], ascending=[True, True])
    # dataset.to_csv("shujutang_VOC_img_filepath.csv", sep='\t', encoding='utf-8')

    set(dataset['class'])
    class_list = list(dataset['class'])
    class_count = {}
    for class1 in tqdm(class_list):
        if class1 not in class_count:
            class_count[class1] = 1
        else:
            class_count[class1] += 1
    class_random_sample_dict = class_count.copy()
    for k, v in class_count.items():
        class_random_sample_dict[k] = round(v * 0.01)

    samples_for_val = []
    for k in class_random_sample_dict:
        # 该类的所有样本组成的列表
        this_class_sample_list = list(dataset[dataset['class'] == k]['path'])
        sample_num = class_random_sample_dict[k]
        samples_for_val += random.sample(this_class_sample_list, sample_num)

    # 使用国家项目的数据线数据集加进去
    teacher_chi_data_base_path = Path("/data/yousixia/images/indoor_new_20220228")
    all_train_val_img_filepaths = [str(pathlib_object) for pathlib_object in teacher_chi_data_base_path.rglob(
        "*")]
    wire_datapaths = []
    for filepath in all_train_val_img_filepaths:
        if 'xml' in filepath and 'wire' in get_name_from_xml(filepath):
            filepath = filepath.replace("xml", "jpg")
            wire_datapaths.append(filepath)
    # 随机取20张
    samples_for_val += random.sample(wire_datapaths, 20)



    for filepath in tqdm(samples_for_val):
        shutil.copy(filepath, "/data/yousixia/images/shujutang_data/shujutang_for_quanzhi_val")
        filepath = filepath.replace("jpg", "xml")
        shutil.copy(filepath, "/data/yousixia/images/shujutang_data/shujutang_for_quanzhi_val")



