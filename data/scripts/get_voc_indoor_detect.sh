#!/usr/bin/env bash

echo "Splitting dataset..."
SRC_DIR="VOCs/indoor_detecting"
DEST_DIR="YOLOs/indoor_detecting"
python3 - "$@" <<END
import xml.etree.ElementTree as ET
# import pickle
import os
# from os import listdir, getcwd
# from os.path import join
import shutil
from multiprocessing.pool import ThreadPool
from tqdm import tqdm


src_dir = "VOCs/indoor_detecting"  ## directory structure must the same as VOC
dest_dir = "YOLOs/indoor_detecting"
# classes = ['feces', 'slippers', 'closetool', 'cabinet', 'curtain', 'trashcan',
#            'stool', 'table', 'bed', 'book', 'socks', 'sofa', 'carpet', 'wire']
classes = []
root = "/home/yousixia/data/"
sets = ['train', 'val', 'test']


def convert(size, box):
	dw = 1. / (size[0])
	dh = 1. / (size[1])
	x = (box[0] + box[1]) / 2.0 - 1
	y = (box[2] + box[3]) / 2.0 - 1
	w = box[1] - box[0]
	h = box[3] - box[2]
	x = x * dw
	w = w * dw
	y = y * dh
	h = h * dh
	return x, y, w, h


def convert_annotation(root_path, image_id):
	# convert VOC xml to yolo txt
	try:
		in_file = open(os.path.join(root_path, src_dir, 'Annotations/%s.xml') % image_id, 'r')
	except FileNotFoundError:
		print('file %s does not exist' % in_file)
		return None
	out_file = open(os.path.join(root_path, dest_dir, 'VOC/labels/%s.txt') % image_id, 'w')
	tree = ET.parse(in_file)
	root = tree.getroot()
	size = root.find('size')
	w = int(size.find('width').text)
	h = int(size.find('height').text)

	for obj in root.iter('object'):
		difficult = obj.find('difficult').text
		cls = obj.find('name').text
		if cls not in classes:
			classes.append(cls)
		if int(difficult) == 1:
			continue
		cls_id = classes.index(cls)
		xmlbox = obj.find('bndbox')
		b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
		     float(xmlbox.find('ymax').text))
		bb = convert((w, h), b)
		out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def merge(file2, file1):
	# merge file2 to file1
	f1 = open(file1, 'a+', encoding='utf-8')
	with open(file2, 'r', encoding='utf-8') as f2:
		for i in f2:
			f1.write(i)
		f1.close()


def format_yolo_data(root, set_name):
	with open(os.path.join(root, dest_dir, '_{}.txt'.format(set_name)), 'r') as f:
		lines = f.readlines()
	os.system('mkdir ' + os.path.join(root, dest_dir, 'VOC/images/{}'.format(set_name)))
	if set_name != 'test':
		os.system('mkdir ' + os.path.join(root, dest_dir, 'VOC/labels/{}'.format(set_name)))
	for line in tqdm(lines, desc=set_name):
		line = "/".join(line.split('/')[-2:]).strip()
		line = os.path.join(root, src_dir, line)
		filename = line.split('/')[-1]
		# 确认是否在
		if os.path.exists(line):
			# 图片文件移到images文件夹
			shutil.copyfile(line, os.path.join(root, dest_dir, "VOC/images/{}".format(set_name), filename))
		# command = "cp " + line + " " + os.path.join(root, dest_dir, "VOC/images/{}".format(set_name))
		# os.system(command)
		if set_name != 'test':
			# 标签文件移到labels文件夹
			line = line.replace('JPEGImages', 'labels')
			line = line.replace(src_dir, dest_dir)
			line = line.replace('jpg', 'txt')
			line = "/".join(line.split('/')[-2:]).strip()
			line = os.path.join(root, dest_dir, 'VOC', line)
			filename = line.split('/')[-1]
			if os.path.exists(line):
				shutil.copyfile(line, os.path.join(root, dest_dir, "VOC/labels/{}".format(set_name), filename))
			# command = "cp " + line + " " + os.path.join(root, dest_dir, "VOC/labels/{}".format(set_name))
			# os.system(command)


if os.path.exists(os.path.join(root, dest_dir)):
	shutil.rmtree(os.path.join(root, dest_dir))

# 从VOC annotations里取出框的数据，转换成yolo所用的txt格式
if not os.path.exists(os.path.join(root, dest_dir, 'VOC/labels/')):
	os.makedirs(os.path.join(root, dest_dir, 'VOC/labels/'))
for image_set in sets:
	image_ids = open(os.path.join(root, src_dir, 'ImageSets/Main/%s.txt') % image_set).read().strip().split()
	list_file = open(os.path.join(root, dest_dir, '_%s.txt') % image_set, 'w')
	for image_id in tqdm(image_ids, desc=image_set):
		list_file.write(os.path.join(root, src_dir, 'JPEGImages/%s.jpg\n') % image_id)
		if image_set != 'test':
			# 不是test，就有xml文件，需要转为txt，存起来
			convert_annotation(root, image_id)
list_file.close()
merge(os.path.join(root, dest_dir, '_val.txt'), os.path.join(root, dest_dir, '_train.txt'))

# 创建存放转换后train和val图片文件的目录
os.system('mkdir -p ' + os.path.join(root, dest_dir, 'VOC/images'))

# format_yolo_data(root, 'train')
# format_yolo_data(root, 'val')
# format_yolo_data(root, 'test')
results = ThreadPool(3).imap(lambda x: format_yolo_data(root, x),
                             ['train', 'val', 'test'])
for result in results:
	if result is not None:
		result.wait()
		result.close()
END

rm -rf /home/yousixia/data/${DEST_DIR}/*.txt # remove temporary _train.txt/_val.txt/_test.txt
echo "amicro data processing done."
