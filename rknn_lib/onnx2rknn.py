# from utils.general import increment_path
import argparse
import os
from pathlib import Path
import random
import shutil
import sys
import numpy as np
import datetime
import yaml
from rknn.api import RKNN

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # current project root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_weights', type=str, default=ROOT / 'yolov5s.onnx', help='initial onnx weights path')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--DATASET', type=str, default=None, help='dataset use to do rknn quantization')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == '__main__':

    # ONNX_MODEL = './runs/train/exp10/weights/best_sim.onnx'
    # RKNN_MODEL = './runs/train/exp10/weights/best_sim_try_single_datasets.rknn'
    # RKNN_PRECOMPILED_MODEL = './runs/train/exp10/weights/yolov5s_precompiled.rknn'
    # DATASET = './rknn/amicro_indoor_try_single/dataset1.txt'
    # ONNX_MODEL = './runs/train/exp10/weights/best_sim.onnx'
    # RKNN_MODEL = './runs/train/exp10/weights/best_sim.rknn'
    # RKNN_PRECOMPILED_MODEL = './runs/train/exp10/weights/yolov5s_precompiled.rknn'
    # DATASET = './rknn/amicro_indoor/amicro_indoor_dataset.txt'
    # ONNX_MODEL = './rknn/yolov5s_rm_transpose.onnx'
    # RKNN_MODEL = './rknn/yolov5s_rm_transpose.rknn'
    # RKNN_PRECOMPILED_MODEL = './runs/train/exp10/weights/yolov5s_precompiled.rknn'
    # DATASET = './rknn/dataset.txt'

    opt = parse_opt()
    # if opt.DATASET is None:
    #     # Read opt.yaml
    #     with open(Path(opt.onnx_weights).parents[1] / "opt.yaml", errors='ignore') as f:
    #         model_training_opt = yaml.safe_load(f)  # dictionary
    #     with open(ROOT / model_training_opt['data'], errors='ignore') as f:
    #         data = yaml.safe_load(f) # dictionary
    #     data_path = data['train']
    #     # ramdom sampling 5000 images to create DATASET.txt
    #     all_imgs = random.sample(os.listdir(data_path), 5000)
    #     opt.DATASET = Path(opt.onnx_weights).parents[1] / "dataset.txt"
    #     if os.path.isfile(opt.DATASET):
    #         print("DATASET exists in model path, directly use it!\n")
    #     else:
    #         f = open(opt.DATASET, "w")
    #         for img in all_imgs:
    #             f.write(str(Path(data_path) / img) + "\n")
    #         f.close()

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                reorder_channel='0 1 2', target_platform='rv1109',
                output_optimize=1)
    print('done')

    print('--> Loading model')
    ret = rknn.load_onnx(model=opt.onnx_weights)
    if ret != 0:
        print('Load onnx model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, pre_compile=False, dataset='dataset.txt')
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    onnx_filename_with_time = str(Path(opt.onnx_weights).stem) + '_' + str(nowTime)
    rknn_model_path = Path(opt.onnx_weights).parent / Path(onnx_filename_with_time).with_suffix('.rknn')
    # assert not rknn_model_path.exists(), f'failed to export rknn file cause it exists: {rknn_model_path}'
    ret = rknn.export_rknn(rknn_model_path)
    if ret != 0:
        print('Export {} failed!'.format(rknn_model_path))
        exit(ret)
    print('done')

    rknn.release()
