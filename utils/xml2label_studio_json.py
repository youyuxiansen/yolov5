import json
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from pathlib import Path


# data_for_save = [{
#   "data": {
#     "image": "/static/samples/sample.jpg"
#   },

#   "annotations": [{
#     "model_version": "one",
#     "score": 0.5,
#     "result": [
#       {
#         "id": "result1",
#         "type": "rectanglelabels",
#         "from_name": "label", "to_name": "image",
#         "original_width": 600, "original_height": 403,
#         "image_rotation": 0,
#         "value": {
#           "rotation": 0,
#           "x": 4.98, "y": 12.82,
#           "width": 32.52, "height": 44.91,
#           "rectanglelabels": ["Airplane"]
#         }
#       },
#       {
#         "id": "result2",
#         "type": "rectanglelabels",
#         "from_name": "label", "to_name": "image",
#         "original_width": 600, "original_height": 403,
#         "image_rotation": 0,
#         "value": {
#           "rotation": 0,
#           "x": 75.47, "y": 82.33,
#           "width": 5.74, "height": 7.40,
#           "rectanglelabels": ["Car"]
#         }
#       },
#       {
#         "id": "result3",
#         "type": "choices",
#         "from_name": "choice", "to_name": "image",
#         "value": {
#           "choices": ["Airbus"]
#       }
#     }]
#   }]
# }]


def write_list_to_json(list, json_file_name, json_file_save_path):
    """
    将list写入到json文件
    :param list:
    :param json_file_name: 写入的json文件名字
    :param json_file_save_path: json文件存储路径
    :return:
    """
    os.chdir(json_file_save_path)
    with open(json_file_name, 'w') as  f:
        f.write(json.dumps(list, ensure_ascii=False))


if __name__ == '__main__':
    base_path = "/data/yousixia/images/shujutang_data/shujutang_to_VOC_resized/"
    all_annotate_files = [pathlib_object for pathlib_object in Path(base_path).rglob(
        "*") if pathlib_object.suffix in ['.xml']]
    data_for_save = []
    for xml in tqdm(all_annotate_files):
        label_studio_dict = {}
        label_studio_dict["data"] = {}
        label_studio_dict["data"]["image"] = "http://192.168.0.175:8081/" + Path(xml).stem + ".jpg"
        label_studio_dict["annotations"] = [{}]
        label_studio_dict["annotations"][0]["model_version"] = "one"
        label_studio_dict["annotations"][0]["score"] = 0.5
        label_studio_dict["annotations"][0]["result"] = []

        xml_info = []
        tree = ET.parse(xml)
        root = tree.getroot()
        for obj in root.iter('size'):
            width = int(obj.find('width').text)
            height = int(obj.find('height').text)
        result_id = 1
        for obj in root.iter('object'):
            result_dict = {}
            cls = obj.find('name').text
            xmin = round(float(obj.find('bndbox').find('xmin').text))
            ymin = round(float(obj.find('bndbox').find('ymin').text))
            xmax = round(float(obj.find('bndbox').find('xmax').text))
            ymax = round(float(obj.find('bndbox').find('ymax').text))
            xml_info.append([str(cls), str(xmin),
                             str(ymin), str(xmax), str(ymax)])
            result_dict["id"] = "result" + str(result_id)
            result_dict["type"] = "rectanglelabels"
            result_dict["from_name"] = "label"
            result_dict["to_name"] = "image"
            result_dict["original_width"] = width
            result_dict["original_height"] = height
            result_dict["image_rotation"] = 0
            result_dict["value"] = {}
            result_dict["value"]["rotation"] = 0
            result_dict["value"]["x"] = round(xmin / width * 100, 2)
            result_dict["value"]["y"] = round(ymin / height * 100, 2)
            result_dict["value"]["width"] = round((xmax - xmin) / width * 100, 2)
            result_dict["value"]["height"] = round((ymax - ymin) / height * 100, 2)
            result_dict["value"]["rectanglelabels"] = [cls]
            label_studio_dict["annotations"][0]["result"].append(result_dict)
            result_id+=1
        data_for_save.append(label_studio_dict)
    write_list_to_json(data_for_save, "xml_to_label_studio_json.json", "/home/yousixia/project/yolov5/")


