import os
from glob import glob
import random
import shutil
from pathlib import Path
from multiprocessing import Process
import xml.etree.ElementTree as ET
from tqdm import tqdm
import cv2
from math import floor
from PIL import Image

import argparse
import pandas as pd
import numpy as np
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


def get_file_name(path):
    base_dir = os.path.dirname(path)
    file_name, ext = os.path.splitext(os.path.basename(path))
    ext = ext.replace(".", "")
    return (base_dir, file_name, ext)


def resize(image_path,
           xml_path,
           newSize,
           output_path,
           mode,
           verbose=False
           ):

    image = cv2.imread(image_path)

    mode = mode and mode.lower()
    # Standard resize mode
    if mode is None or mode == 'size':
        newSize = (int(newSize[0]), int(newSize[1]))
        scale_x = float(newSize[0]) / float(image.shape[1])
        scale_y = float(newSize[1]) / float(image.shape[0])
        image = cv2.resize(src=image, dsize=(newSize[0], newSize[1]))
    else:
        # Scaling by factor or percentage of the original image size
        if mode == 'scale' or mode == 'percentage':
            mul = 0.01 if mode == 'percentage' else 1.0
            newSize = (
                floor(float(image.shape[1]) * float(newSize[0]) * mul),
                floor(float(image.shape[0]) * float(newSize[1]) * mul))
            scale_x = newSize[0] / image.shape[1]
            scale_y = newSize[1] / image.shape[0]
            interp = cv2.INTER_LINEAR if (scale_x > 1.0 or scale_y > 1.0) else cv2.INTER_AREA
            image = cv2.resize(
                src=image,
                dsize=(0, 0), dst=None,
                fx=scale_x, fy=scale_y, interpolation=interp)
        # Target mode; choose the correct ratio to reach one of the x/y targets without oversize
        elif mode == 'target':
            ratio = float(int(newSize[0])) / float(image.shape[1])
            targetRatio = float(int(newSize[1])) / float(image.shape[0])
            ratio = targetRatio if targetRatio < ratio else ratio
            scale_x = scale_y = ratio
            interp = cv2.INTER_LINEAR if (scale_x > 1.0 or scale_y > 1.0) else cv2.INTER_AREA
            image = cv2.resize(
                src=image,
                dsize=(0, 0), dst=None,
                fx=scale_x, fy=scale_y, interpolation=interp)
        else:
            raise Exception(f"Invalid resize mode: {mode}")

    newBoxes = []
    xmlRoot = ET.parse(xml_path).getroot()
    xmlRoot.find('filename').text = image_path.split('/')[-1]
    size_node = xmlRoot.find('size')
    size_node.find('width').text = str(newSize[0])
    size_node.find('height').text = str(newSize[1])

    for member in xmlRoot.findall('object'):
        bndbox = member.find('bndbox')

        xmin = bndbox.find('xmin')
        ymin = bndbox.find('ymin')
        xmax = bndbox.find('xmax')
        ymax = bndbox.find('ymax')

        xmin.text = str(int(np.round(float(xmin.text) * scale_x)))
        ymin.text = str(int(np.round(float(ymin.text) * scale_y)))
        xmax.text = str(int(np.round(float(xmax.text) * scale_x)))
        ymax.text = str(int(np.round(float(ymax.text) * scale_y)))

    (_, file_name, ext) = get_file_name(image_path)
    cv2.imwrite(os.path.join(output_path, '.'.join([file_name, ext])), image)

    tree = ET.ElementTree(xmlRoot)
    tree.write('{}/{}.xml'.format(output_path, file_name), encoding="UTF-8")


def process_image(file_path, output_path, x, y, mode):
    # from https://github.com/italojs/resize_dataset_pascalvoc/
    (base_dir, file_name, ext) = get_file_name(file_path)
    image_path = '{}/{}.{}'.format(base_dir, file_name, ext)
    xml = '{}/{}.xml'.format(base_dir, file_name)
    try:
        resize(
            image_path,
            xml,
            (x, y),
            output_path,
            mode,
        )
    except Exception as e:
        print('[ERROR] error with {}\n file: {}'.format(image_path, e))
        print('--------------------------------------------------')


from multiprocessing import Pool
from multiprocessing import freeze_support
def do_resize_with_concurrent(image_path):
    # for image_path in tqdm(image_path):
    try:
        process_image(image_path, str(Path(image_path).parent) + "_resized", 640, 640, mode='size')
    except:
        print("error with item")


if __name__ == '__main__':
    # --data-root /home/yousixia/data/images
    parser = argparse.ArgumentParser(
        description='This program create '
                    'the directory structure like VOC and used to train yolo model')
    parser.add_argument('--data-root', "-r", type=str, default='/home/yousixia/data/images',
                        help='Directory for saving frame images')
    opt = parser.parse_args()
    opt.data_root = Path(opt.data_root).as_posix()

    global all_xml_files
    global all_img_files
    base_path = Path(opt.data_root)
    all_files = [pathlib_object for pathlib_object in base_path.rglob(
        "*") if pathlib_object.suffix in ['.jpg', '.png', '.xml']]
    all_xml_files = [x for x in all_files if x.suffix == '.xml']
    all_img_files = [x for x in all_files if (x.suffix in ['.png', '.jpg']) and (
        '门槛' not in x.stem and '门-门框' not in x.stem and '门' not in x.stem)]

    global all_xml_stem_files
    global all_img_stem_files
    global all_xml_str_files
    global all_img_str_files
    global all_xml_files_str
    global all_img_files_str

    all_xml_stem_files = [x.stem for x in all_xml_files]
    all_img_stem_files = [x.stem for x in all_img_files]
    all_xml_str_files = [str(x) for x in all_xml_files]
    all_img_str_files = [str(x) for x in all_img_files]
    # all_img_stem_files = set(all_img_stem_files)

    def get_repeat_file():
        # 提取重复文件到txt中，反馈给数据堂
        def write_rep_filepath_to_txt(all_stem_files, all_str_files, output_filename):
            f = open(output_filename + ".txt", "w")
            save_dict = {}
            rep_str_filepath = []
            for i, x in enumerate(tqdm(all_stem_files)):
                if x not in save_dict:
                    save_dict[x] = 1
                elif save_dict[x] == 1:
                    save_dict[x] += 1
                elif save_dict[x] > 1:
                    f.write(all_str_files[i] + '\n')
                    rep_str_filepath.append(all_str_files[i])
            f.close()
        img_p = Process(target=write_rep_filepath_to_txt, args=(
            all_img_stem_files, all_img_str_files, "img_rep_filepath", ))
        xml_p = Process(target=write_rep_filepath_to_txt, args=(
            all_xml_stem_files, all_xml_str_files, "xml_rep_filepath", ))
        img_p.start()
        xml_p.start()
        img_p.join()
        xml_p.join()

    shujutang_to_VOC_base_path = Path("/data/yousixia/images/shujutang_data/wire_to_VOC")
    shujutang_to_VOC_all_files = [pathlib_object for pathlib_object in shujutang_to_VOC_base_path.rglob(
        "*") if pathlib_object.suffix in ['.jpg', '.xml']]
    all_xml_files = [x for x in shujutang_to_VOC_all_files if x.suffix == '.xml']
    all_img_files = [x for x in shujutang_to_VOC_all_files if (x.suffix in ['.png', '.jpg']) and (
        '门槛' not in x.stem and '门-门框' not in x.stem and '门' not in x.stem)]
    all_img_stem_files = [x.stem for x in all_xml_files]
    all_img_filesall_img_files_str_stem = [x.stem for x in all_img_files]
    all_xml_files_str = [str(x) for x in all_xml_files]
    all_img_files_str = [str(x) for x in all_img_files]

    def get_label_missed_sample():
        # 移动遗漏标注的图片到另一个文件夹
        image_missing_stem = list(set(all_xml_stem_files).difference(set(all_img_stem_files)))
        xml_missing_stem = list(set(all_img_stem_files).difference(set(all_xml_stem_files)))
        global all_img_files
        global all_xml_files
        global all_xml_files_str
        all_not_labeled_img_files = [x for x in all_img_files if x.stem in image_missing_stem]
        for x in all_not_labeled_img_files:
            shutil.move(str(x), "/data/yousixia/images/shujutang_data/遗漏标注图片")

        # 核验是否有遗漏的xml和jpg文件
        all_xml_files_filename = [x.stem + x.suffix for x in all_xml_files]
        all_img_files_filename = [x.stem + x.suffix for x in all_img_files]

        print(list(set(all_img_stem_files) ^ set(all_img_stem_files)))
        all_files_str = all_xml_files_filename + all_img_files_filename
        shujutang_to_VOC_all_files_stem = [x.stem + x.suffix for x in shujutang_to_VOC_all_files]
        print(list(set(all_files_str) ^ set(shujutang_to_VOC_all_files_stem)))

    def move_to_VOC():
        # 复制所有已标注xml和jpg到另一个文件夹下，以便复用构造VOC目录结构的脚本
        def data_copy_to_new_folder(filepaths, folder):
            for x in tqdm(filepaths):
                shutil.copy(str(x), folder)

        img_p = Process(target=data_copy_to_new_folder, args=(
            all_img_str_files, str(shujutang_to_VOC_base_path), ))
        xml_p = Process(target=data_copy_to_new_folder, args=(
            all_xml_str_files, str(shujutang_to_VOC_base_path), ))
        img_p.start()
        xml_p.start()
        img_p.join()
        xml_p.join()

    def get_size_unmatched_img_and_xml():
        # 输出xml中图片尺寸和图片实际尺寸不符的图片和xml文件
        shape_not_right_folder = "/data/yousixia/images/shujutang_data/shape_not_right_data"
        shape_not_right_xml = []
        shape_not_right_img = []
        global all_img_stem_files
        global all_img_stem_files
        global all_xml_files_str
        global all_img_files_str
        if os.path.exists(shape_not_right_folder):
            shutil.rmtree(shape_not_right_folder)
        os.makedirs(shape_not_right_folder)
        all_xml_files_str = sorted(all_xml_files_str)
        all_img_files_str = sorted(all_img_files_str)
        print(list(set(all_img_stem_files) ^ set(all_img_stem_files)))
        no_object_in_img = list(set(all_img_stem_files) ^ set(all_img_stem_files))

        img_increment_id = 0
        original_img_folder = "/data/yousixia/images/shujutang_data/购买数据/数据线22077张/数据线22077张"
        for i in tqdm(range(len(all_xml_files_str))):
            while Path(all_xml_files_str[i]).stem != Path(all_img_files_str[i + img_increment_id]).stem:
                print("{} different with {}".format(all_xml_files_str[i], Path(all_img_files_str[i + img_increment_id])))
                shutil.copy(Path(original_img_folder) / (Path(all_img_files_str[i + img_increment_id]).stem + Path(all_img_files_str[i + img_increment_id]).suffix), str(Path(all_img_files_str[i + img_increment_id]).parent))
                img_increment_id += 1
            tree = ET.parse(all_xml_files_str[i])
            root = tree.getroot()
            for size in root.iter('size'):
                xml_shape = (int(size.find('width').text), int(size.find('height').text))
            # img = cv2.imread(str(all_img_files_str[i]))

            img_found = 0
            while not img_found:
                try:
                    with Image.open(str(all_img_files_str[i])) as img:
                        img_shape = img.size
                        img_found = 1
                except:
                    print("{} is Broken".format(Path(all_img_files_str[i]).stem))
                    img_found = 0
                    shutil.copy(Path(original_img_folder) / (Path(all_img_files_str[i]).stem + Path(all_img_files_str[i]).suffix), str(Path(all_img_files_str[i]).parent))
            # img = magic.from_file(str(all_img_files_str[i]))
            # img_shape = (img.shape[1], img.shape[0])
            # img_shape_tuple = re.search('(\d+)x(\d+)', img).groups()
            # img_shape = tuple([int(x) for x in img_shape_tuple])
            if xml_shape != img_shape:
                shape_not_right_xml.append(str(all_xml_files_str[i]))
                shape_not_right_img.append(str(all_img_files_str[i]))
                shutil.copy(str(all_xml_files_str[i]), shape_not_right_folder)
                shutil.copy(str(all_img_files_str[i]), shape_not_right_folder)

    def resize_all_img_and_xml():
        freeze_support()

        # v3
        def run_multiprocessing(func, i, n_processors):
            with Pool(processes=n_processors) as pool:
                return pool.map(func, tqdm(i))
        n_processors = 20
        resized_dir = "/data/yousixia/images/shujutang_data/wire_to_VOC_resized/"
        if os.path.exists(resized_dir):
            shutil.rmtree(resized_dir)
        os.makedirs(resized_dir)
        x_ls = all_img_str_files
        base_path_for_mp = [str(shujutang_to_VOC_base_path)] * len(all_img_str_files)
        out = run_multiprocessing(do_resize_with_concurrent, x_ls, n_processors)

        # v2
        # import concurrent.futures
        # from more_itertools import grouper
        # executor = concurrent.futures.ProcessPoolExecutor(10)
        # futures = [executor.submit(do_resize_with_concurrent, item).result() for item in all_img_str_files]
        # concurrent.futures.wait(futures)

        # v1
        # futures = [executor.submit(do_resize_with_concurrent, group)
        #     for group in grouper(all_img_str_files, 30000)]
        # concurrent.futures.wait(futures)

    def extract_some_to_do_val():
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
        include_classes = list('污渍', '宠物粪便', '鞋子', '袜-布', '袜子', '鞋', 'IMG', '地垫', '垃圾桶', '纸团', '风扇底座')
        class_list = list(dataset['class'])
        class_count = {}
        for class1 in tqdm(class_list):
            if class1 in include_classes:
                if class1 not in class_count:
                    class_count[class1] = 1
                else:
                    class_count[class1] += 1
        class_random_sample_dict = class_count.copy()
        for k, v in class_count.items():
            if k == "IMG":
                # IMG开头代表数据线
                class_random_sample_dict[k] = v
                continue
            class_random_sample_dict[k] = round(v * 0.01)

        samples_for_val = []
        for k in class_random_sample_dict:
            # 该类的所有样本组成的列表
            this_class_sample_list = list(dataset[dataset['class'] == k]['path'])
            sample_num = class_random_sample_dict[k]
            samples_for_val += random.sample(this_class_sample_list, sample_num)

        # 使用国家项目的数据线数据集加进去
        # teacher_chi_data_base_path = Path("/data/yousixia/images/indoor_new_20220228")
        # all_train_val_img_filepaths = [str(pathlib_object) for pathlib_object in teacher_chi_data_base_path.rglob(
        #     "*")]
        # wire_datapaths = []
        # for filepath in all_train_val_img_filepaths:
        #     if 'xml' in filepath and 'wire' in get_name_from_xml(filepath):
        #         filepath = filepath.replace("xml", "jpg")
        #         wire_datapaths.append(filepath)
        # # 随机取20张
        # samples_for_val += random.sample(wire_datapaths, 20)

        # 第一列的中文是{'袜子', '垃圾桶', '纸团', '风扇底座', '鞋', '污渍', '鞋子', '宠物粪便', '地垫', '体重秤', '袜-布'}
        samples_for_val_filenames = pd.DataFrame(samples_for_val)[0].str.rsplit("/", n=1, expand=True)[1]
        chinese_in_first_col_to_set = set(
            [x for x in samples_for_val_filenames.str.split('\_', expand=True)[0] if '.jpg' not in x])
        print(chinese_in_first_col_to_set)
        # 第二列的中文是污渍-({'酱油或醋', '咖啡', '牛奶', '粥', '茶'})
        chinese_in_second_col_to_set = set([x for x in samples_for_val_filenames.str.split(
            '\_', expand=True)[1] if x is not None and 'house' not in x and '.jpg' not in x])
        print(chinese_in_second_col_to_set)
        # flavor
        replece_chinese_dict = {
            '袜子': 'socks',
            '垃圾桶': 'trashcan',
            '纸团': 'paper',
            '风扇底座': 'fanbase',
            '污渍': {
                '茶': 'tea',
                '牛奶': 'milk',
                '咖啡': 'coffee',
                '酱油或醋': 'flavor',
                '粥': 'congee',
            },
            '鞋子': 'shoes',
            '子': '',
            '鞋': 'shoes',
            '宠物粪便': 'feces',
            '地垫': 'carpet',
            '体重秤': 'scale',
            '袜-布': 'clothes'
        }

        for filepath in tqdm(samples_for_val):
            english_name = ''
            # 将文件名中出现的中文改为英文
            old_name = Path(filepath).stem + Path(filepath).suffix
            for key in replece_chinese_dict:
                if key in old_name:
                    if '污渍' in old_name:
                        for key1 in replece_chinese_dict['污渍']:
                            if key1 in old_name:
                                english_name = old_name.replace('污渍_' + key1, replece_chinese_dict['污渍'][key1])
                                break
                        break
                    english_name = old_name.replace(key, replece_chinese_dict[key])
                    break
            shutil.copy(filepath, "/data/yousixia/images/shujutang_data/shujutang_for_quanzhi_val/" + english_name)
            filepath = filepath.replace("jpg", "xml")
            english_name = english_name.replace("jpg", "xml")
            shutil.copy(filepath, "/data/yousixia/images/shujutang_data/shujutang_for_quanzhi_val/" + english_name)


    # get_repeat_file()
    # get_label_missed_sample()
    # move_to_VOC()
    get_size_unmatched_img_and_xml()
    resize_all_img_and_xml()
    # extract_some_to_do_val()
