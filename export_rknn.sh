#! /bin/bash

MODELFILE=runs/train/indoor/exp/weights/best.pt
# Delete the character after the last slash.
JUSTPATH=$PWD/${MODELFILE%/*}
# remove the suffix .pt
a=(${MODELFILE//.pt/ })
# FILE_WITHOUT_SUFFIX=yolov5s
FILE_WITHOUT_SUFFIX=${a##*/}

/home/yousixia/anaconda3/envs/tensorrt_yolov3/bin/python export.py --weights "${JUSTPATH}"/"${FILE_WITHOUT_SUFFIX}".pt --simplify --opset 10
/home/yousixia/anaconda3/envs/rknn/bin/python rknn/onnx2rknn.py --onnx_weights "${JUSTPATH}"/"${FILE_WITHOUT_SUFFIX}".onnx \
 --DATASET "${PWD}"/rknn/amicro_indoor/amicro_indoor_dataset.txt