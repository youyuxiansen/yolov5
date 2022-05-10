from time import sleep
from rknn_impl import RknnImpl
from yolox_detector_impl import AmicroIndoorYoloxDetector
# from head_detector_impl import HeadDetector

if __name__ == '__main__':
    imgPath = "data/images/IMG_20220505_105041.jpg"

    # 原始yolo算法示例
    MODEL_PATH = "yolox_2022-05-06-14-36-43_precompiled.rknn"
    PARAMS = {"mean_values":[[0.]*3], "std_values":[[255.0]*3], "reorder_channel":'0 1 2'}

    nn = RknnImpl(**PARAMS)
    detector = AmicroIndoorYoloxDetector(nn)
    # exit(0)

    detector.init(MODEL_PATH)
    detector.detect_suit_yolo(imgPath)


