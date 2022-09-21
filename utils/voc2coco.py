import os
import argparse
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from typing import Dict, List
from tqdm import tqdm
import re


def get_label2id(labels_path: str) -> Dict[str, int]:
    """id is 1 start"""
    with open(labels_path, 'r') as f:
        labels_str = f.read().split()
    labels_ids = list(range(1, len(labels_str) + 1))
    return dict(zip(labels_str, labels_ids))


def get_annpaths(ann_dir_path: str = None,
                 train_ann_ids_path: str = None,
                 val_ann_ids_path: str = None,
                 ext: str = '') -> List[str]:

    # If use annotaion ids list
    ext_with_dot = '.' + ext if ext != '' else ''
    with open(train_ann_ids_path, 'r') as f:
        ann_ids = f.read().split()
    train_ann_paths = [os.path.join(ann_dir_path, aid + ext_with_dot) for aid in ann_ids]
    with open(val_ann_ids_path, 'r') as f:
        ann_ids = f.read().split()
    val_ann_paths = [os.path.join(ann_dir_path, aid + ext_with_dot) for aid in ann_ids]
    return train_ann_paths, val_ann_paths


def get_image_info(annotation_root, image_path, extract_num_from_imgid=True):
    # path = annotation_root.findtext('path')
    path = None
    if path is None:
        # Assume all image is jpg! TODO get true image suffix for support different image format
        filename = Path(image_path).stem + ".jpg"
    else:
        filename = os.path.basename(path)
    img_name = os.path.basename(filename)
    img_id = os.path.splitext(img_name)[0]
    if extract_num_from_imgid and isinstance(img_id, str):
        img_id = int(re.findall(r'\d+', img_id)[0])

    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info


def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext('name')
    assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.findtext('xmin')) - 1
    ymin = int(bndbox.findtext('ymin')) - 1
    xmax = int(bndbox.findtext('xmax'))
    ymax = int(bndbox.findtext('ymax'))
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        # 'area': o_width * o_height,
        # 'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        # 'ignore': 0,
        # 'segmentation': []  # This script is not for segmentation
    }
    return ann


def convert_xmls_to_cocojson(annotation_paths: List[str],
                             label2id: Dict[str, int],
                             output_jsonpath: str,
                             extract_num_from_imgid: bool = True):
    output_json_dict = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?
    global image_id
    print('Start converting !')
    for a_path in tqdm(annotation_paths):
        # Read annotation xml
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()

        img_info = get_image_info(annotation_root=ann_root,
                                  image_path=a_path,
                                  extract_num_from_imgid=extract_num_from_imgid)
        img_info['id'] = image_id
        output_json_dict['images'].append(img_info)

        for obj in ann_root.findall('object'):
            ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
            ann.update({'image_id': image_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

        image_id += 1

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    with open(output_jsonpath, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)


def move_img(output_path: str, img_path: str, annotation_paths: List[str], image_folder_name: str):
    new_img_path = os.path.join(output_path, image_folder_name)
    os.makedirs(new_img_path)
    image_paths = os.listdir(img_path)
    image_paths = [x for x in image_paths if Path(x).suffix in ['.jpg', '.png', '.jpeg']]
    anna_stem_list = [Path(x).stem for x in annotation_paths]
    for img in tqdm(image_paths):
        Path_img = Path(img)
        if Path_img.stem in anna_stem_list:
            shutil.copy(os.path.join(img_path, img), os.path.join(new_img_path, Path_img.stem + Path_img.suffix))



def main():
    parser = argparse.ArgumentParser(
        description='This script support converting voc format xmls to coco format json')
    parser.add_argument('--ann_dir', type=str, default=None,
                        help='path to annotation files directory.')
    parser.add_argument('--img_dir', type=str, default=None,
                        help='path to image files directory.')
    parser.add_argument('--train_ann_ids', type=str, default=None,
                        help='path to train annotation files ids list.')
    parser.add_argument('--val_ann_ids', type=str, default=None,
                        help='path to val annotation files ids list.')
    parser.add_argument('--labels', type=str, default=None,
                        help='path to label list.')
    parser.add_argument('--output_dir', type=str, default=None, help='path to coco format data\'s output dir')
    parser.add_argument('--ext', type=str, default='', help='additional extension of annotation file')
    args = parser.parse_args()
    label2id = get_label2id(labels_path=args.labels)

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    train_ann_paths, val_ann_paths = get_annpaths(
        ann_dir_path=args.ann_dir,
        train_ann_ids_path=args.train_ann_ids,
        val_ann_ids_path=args.val_ann_ids,
        ext=args.ext
    )
    global image_id
    image_id = 1
    convert_xmls_to_cocojson(
        annotation_paths=train_ann_paths,
        label2id=label2id,
        output_jsonpath=os.path.join(args.output_dir, "train.json"),
        extract_num_from_imgid=True
    )
    convert_xmls_to_cocojson(
        annotation_paths=val_ann_paths,
        label2id=label2id,
        output_jsonpath=os.path.join(args.output_dir, "val.json"),
        extract_num_from_imgid=True
    )
    move_img(
        output_path=args.output_dir,
        img_path=args.img_dir,
        annotation_paths=train_ann_paths,
        image_folder_name="train"
    )
    move_img(
        output_path=args.output_dir,
        img_path=args.img_dir,
        annotation_paths=val_ann_paths,
        image_folder_name="val"
    )


if __name__ == '__main__':
    main()
