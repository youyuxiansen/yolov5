import os
from glob import glob
import shutil
from pathlib import Path
import platform
import time
from tqdm import tqdm

import argparse
from sklearn.model_selection import train_test_split


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
        for xml_file in tqdm(xml_files):
            shutil.copy(xml_file, saved_path / "Annotations/")
        xml_files = path.glob("*.xml")
        xml_stem_list = [Path(x).stem for x in xml_files]
        img_stem_list = [x.stem for x in list(image_files)]
        xml_missing_list = list(set(img_stem_list).difference(set(xml_stem_list)))
        for image in tqdm(image_files):
            if not has_hard_sample:
                # check if pic has a xml correlate to it.
                if image.stem not in xml_missing_list:
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
    trainval_files = [i.split("/")[-1].split(".xml")[0] for i in tqdm(trainval_files)]
    total_files = os.listdir(saved_path / "JPEGImages")
    total_files = sorted(total_files)
    trainval_files = sorted(trainval_files)
    total_files = [Path(x).stem for x in total_files]
    test_files = list(set(total_files).difference(set(trainval_files)))
    for file in tqdm(trainval_files):
        ftrainval.write(file + "\n")
    # test
    for file in tqdm(test_files):
        ftest.write(file + "\n")
    # split
    train_files, val_files = train_test_split(trainval_files, test_size=0.15, random_state=42)
    # train
    for file in tqdm(train_files):
        ftrain.write(file + "\n")

    # val
    for file in tqdm(val_files):
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
                             'it should locates under "/home/yousixia/data/VOCs"（e.q. /data/yousixia/VOCs/VOC2007_amicro）')
    parser.add_argument('--has-hard-sample', "-hard", default=True, action='store_false',
                        help="Include the images in folder which don't have correlated xml files")
    # parser.add_argument('--label-path', nargs='+', type=str, help='Directory containing xml labels')
    opt = parser.parse_args()
    opt.data_root = Path(opt.data_root).as_posix()

    if os.path.isdir(opt.target_folder_path):
        raise Exception('target-folder-path: "{}" existed, please remove it manually first.'.format(opt.target_folder_path))


    # no commas means only one folder
    if ',' in opt.images_folder_name:
        all_image_folder = opt.images_folder_name.split(',')
        # delete all empty string
        all_image_folder = [x for x in all_image_folder if x != '']
    else:
        all_image_folder = [opt.images_folder_name]
    labelme_path = [Path(opt.data_root) / x for x in
                    all_image_folder]  # 原始labelme标注数据路径，文件夹里包含jpg和xml文件，negative_image文件夹仅包含jpg，放加强训练用的负样本。

    construct_amicro_img_dir(opt.target_folder_path,
                             labelme_path, opt.has_hard_sample)
