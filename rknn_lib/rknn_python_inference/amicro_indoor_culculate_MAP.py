from time import sleep
from rknn_impl import RknnImpl
from yolov5_detector_impl import FaceYolov5Detector, OriginYolov5Detector
# from head_detector_impl import HeadDetector

if __name__ == '__main__':
    imgPath = "../data/images/bus.jpg"

    # 原始yolo算法示例
    MODEL_PATH = "../model/best_rknn_20211202.onnx"
    PARAMS = {"mean_values":[[0.]*3], "std_values":[[255.0]*3], "reorder_channel":'0 1 2'}

    nn = RknnImpl(**PARAMS)
    detector = OriginYolov5Detector(nn)
    # exit(0)

    detector.init(MODEL_PATH)
    detector.detect(imgPath)

    exit(0)
    # 人脸检测
    # 一开始没使用工具转换工具，在前处理的中对Slice进行了支持
    MODEL_PATH = "../model/face-focus-removed.onnx"
    PARAMS = {"mean_values":[[0.] * 12], "std_values":[[255.0] * 12], "reorder_channel":''}

    # # 使用了可转换工具，commons中都是转换rknn的工具
    MODEL_PATH = "../model/face-model.onnx"
    PARAMS = {"mean_values":[[0.] * 3], "std_values":[[255.0] * 3], "reorder_channel":'0 1 2'}

    nn = RknnImpl(**PARAMS)
    detector = FaceYolov5Detector(nn)
    detector.init(MODEL_PATH)
    detector.detect(imgPath)
