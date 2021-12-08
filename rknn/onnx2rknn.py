import argparse
import os
from pathlib import Path
import sys
import numpy as np
from rknn.api import RKNN

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_weights', type=str, default=ROOT / 'yolov5s.onnx', help='initial onnx weights path')
    parser.add_argument('--DATASET', type=str, help='dataset use to do rknn quantization')

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
    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                reorder_channel='0 1 2', target_platform='rv1109', output_optimize=1)
    print('done')

    print('--> Loading model')
    ret = rknn.load_onnx(model=opt.onnx_weights)
    if ret != 0:
        print('Load onnx model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='./rknn/dataset.txt')
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    rknn_model_path = Path(opt.onnx_weights).with_suffix('.rknn')
    assert not rknn_model_path.exists(), f'failed to export rknn file cause it exists: {rknn_model_path}'
    ret = rknn.export_rknn(rknn_model_path)
    if ret != 0:
        print('Export {} failed!'.format(rknn_model_path))
        exit(ret)
    print('done')

    # ret = rknn.init_runtime(target='rv1109', rknn2precompile=True)
    # if ret != 0:
    #     print('Init runtime environment failed')
    #     exit(ret)
    # print('done')

    # ret = rknn.export_rknn_precompile_model(RKNN_PRECOMPILED_MODEL)

    rknn.release()
