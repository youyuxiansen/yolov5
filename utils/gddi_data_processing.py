import cv2
from pathlib import Path
import shutil
import os
import xml.etree.ElementTree as ET
from math import floor
import numpy as np
from tqdm import tqdm


def get_file_name(path):
    base_dir = os.path.dirname(path)
    file_name, ext = os.path.splitext(os.path.basename(path))
    ext = ext.replace(".", "")
    return (base_dir, file_name, ext)


def draw_box(boxes, image, path):
    for i in range(0, len(boxes)):
        cv2.rectangle(image, (boxes[i][2], boxes[i][3]), (boxes[i][4], boxes[i][5]), (255, 0, 0), 3)
    cv2.imwrite(path, image)


def resize(image_path,
           xml_path,
           newSize,
           output_path,
           mode,
           save_box_images=False):

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
    size_node.find('width').text = str(image.shape[1])
    size_node.find('height').text = str(image.shape[0])
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

        newBoxes.append([
            1,
            0,
            int(float(xmin.text)),
            int(float(ymin.text)),
            int(float(xmax.text)),
            int(float(ymax.text))
        ])

    (_, file_name, ext) = get_file_name(image_path)
    cv2.imwrite(os.path.join(output_path, '.'.join([file_name, ext])), image)

    tree = ET.ElementTree(xmlRoot)
    tree.write('{}/{}.xml'.format(output_path, file_name, ext))
    if int(save_box_images):
        save_path = '{}/boxes_images/boxed_{}'.format(output_path, ''.join([file_name, '.', ext]))
        if not os.path.exists(Path(save_path).parent):
            os.makedirs(Path(save_path).parent)
    draw_box(newBoxes, image, save_path)


if __name__ == '__main__':
    data_root = '/data/yousixia/images/val_for_gddi/val'
    base_path = Path(data_root)
    all_files = [pathlib_object for pathlib_object in base_path.glob(
        "*") if pathlib_object.suffix in ['.jpg', '.png', '.xml']]
    all_xml_files = [x for x in all_files if x.suffix == '.xml']
    all_img_files = [x for x in all_files if (x.suffix in ['.png', '.jpg']) and (
        '门槛' not in x.stem and '门-门框' not in x.stem and '门' not in x.stem)]
    all_xml_stem_files = [x.stem for x in all_xml_files]
    all_img_stem_files = [x.stem for x in all_img_files]
    not_labeled_stem = list(set(all_xml_stem_files) ^ set(all_img_stem_files))
    for x in not_labeled_stem:
        os.remove(os.path.join(data_root, x + '.xml'))

    for x in tqdm(all_img_files):
        resize(str(x),
               os.path.join(data_root, x.stem + '.xml'),
               (1920, 1080),
               str(x.parent),
               'target',
               save_box_images=True)
