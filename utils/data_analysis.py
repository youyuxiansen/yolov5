from collections import Counter
import os
import argparse
import shutil
from tqdm import tqdm
import xml.etree.ElementTree as ET
from config.cls_name_replacement_config import replace_relation, shujut_relation
from pathlib import Path
from typing import List
from support_img_type import support_img_type


def get_name_from_xml(xml_file: str) -> list:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    cls_list = []
    cls_with_imagenames = {}
    for obj in root.iter('object'):
        cls = obj.find('name').text
        cls_list.append(cls)
    return cls_list


def replace_cls_name_xml(xml_file: str) -> None:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for obj in root.iter('object'):
        try:
            # print(obj.find('name').text, '\n', xml_file)
            if opt.replace_clsname:
                obj.find('name').text = obj.find('name').text.replace(
                    obj.find('name').text, shujut_relation[obj.find('name').text])
        except (AttributeError, KeyError):
            pass
        tree.write(xml_file, encoding='utf-8')


moved_files = []
mkdir_flag = False
def remove_specfied_class_name_file(anno_dir: str, xml: str, class_names: List[str], useless_cls_list: List[str]) -> None:
    global mkdir_flag, moved_files
    bak_dir = Path(anno_dir).parent / (Path(anno_dir).stem + '_bak')
    # if not mkdir_flag:
    #     if os.path.exists(bak_dir):
    #         os.rmdir(bak_dir)
    #     bak_dir.mkdir(parents=True)
    #     mkdir_flag = True
    if len([i for i in useless_cls_list if i in class_names]) > 0:
        # remove xml
        moved_files.append(xml)
        shutil.copy(anno_dir / Path(xml), bak_dir / xml)
        tree = ET.parse(anno_dir / Path(xml))
        root = tree.getroot()
        for obj in root.iter('object'):
            if obj.find('name').text in useless_cls_list:
                root.remove(obj)
        tree.write(anno_dir / Path(xml), encoding='utf-8')
        print("change useless cls in annotate file %s".format(str(anno_dir / Path(xml))))
        # Remove img
        if 0:
            for suffix in support_img_type:
                filename_convert2_img = xml
                filename_convert2_img = filename_convert2_img.replace(".xml", suffix)
                if os.path.exists(Path(anno_dir) / filename_convert2_img):
                    print("moved useless file %s to %s".format(str(filename_convert2_img), str(bak_dir / filename_convert2_img)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annatation_dir', '-a', type=str, help='Directory containing annatation files')
    # parser.add_argument('--image-dir', '-i', type=str, help='Directory containing images')
    parser.add_argument('--print_cls', '-pc', action='store_false',
                        help='Print unique class names')
    parser.add_argument('--replace_clsname', '-r', action='store_true',
                        help='Specify config file for replacing class name in annatation')
    parser.add_argument('--remove_useless_cls', '-m', action='store_true',
                        help='Remove useless cls in annotates(not train with these cls).')
    # TODO: needs support coco json files
    # parser.add_argument('--annatation_type', type=str, help='xml means VOC;json means coco')
    opt = parser.parse_args()
    xml_files = os.listdir(opt.annatation_dir)
    xml_files = [x for x in xml_files if x.endswith('.xml') ]

    if opt.print_cls:
        allcls_list = []
        object_greater_than_2 = []
        allcls_with_imagenames = {}
        useless_cls = ["doorsill", "doorframe"]
        for xml in tqdm(xml_files):
            cls_list = get_name_from_xml(os.path.join(opt.annatation_dir, xml))
            if opt.remove_useless_cls:
                remove_specfied_class_name_file(opt.annatation_dir, xml, cls_list, useless_cls)
            if len(set(cls_list)) > 1:
                object_greater_than_2.append(os.path.join(opt.annatation_dir, xml))
            allcls_list += cls_list

        print(set(allcls_list), '\n')
        print(Counter(allcls_list), '\n')
        # print(object_greater_than_2, '\n')

    if opt.replace_clsname:
        for xml in tqdm(xml_files):
            replace_cls_name_xml(os.path.join(opt.annatation_dir, xml))
        allcls_list = []
        object_greater_than_2 = []
        allcls_with_imagenames = {}
        for xml in tqdm(xml_files):
            cls_list = get_name_from_xml(os.path.join(opt.annatation_dir, xml))

            if len(set(cls_list)) > 1:
                object_greater_than_2.append(os.path.join(opt.annatation_dir, xml))
            allcls_list += cls_list

        print("After replace_clsname: \n")
        print(set(allcls_list), '\n')
        print(Counter(allcls_list), '\n')
        # print(object_greater_than_2, '\n')


