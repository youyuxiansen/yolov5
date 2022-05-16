from cv2 import imwrite
from rknn_lib.rknn_python_inference.detector_bridge import Detector
import cv2
import numpy as np
import random
import torch
import sys
import os
from pathlib import Path


def get_max_scale(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1)
    return scale


def get_new_size(img, scale):
    return tuple(map(int, np.array(img.shape[:2][::-1]) * scale))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class AutoScale:
    def __init__(self, img, max_w, max_h):
        self._src_img = img
        self.scale = get_max_scale(img, max_w, max_h)
        self._new_size = get_new_size(img, self.scale)
        self.__new_img = None

    @property
    def size(self):
        return self._new_size

    @property
    def new_img(self):
        if self.__new_img is None:
            self.__new_img = cv2.resize(self._src_img, self._new_size)
        return self.__new_img


# 前处理
# 1. 缩放640
# 2. 补边
"""
yolov5算法基类
"""


class BaseYOLOXDetector(Detector):
    _masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    _conf_thresh = 0.4
    _iou_thresh = 0.45
    _names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
              'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
              'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
              'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
              'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
              'hair drier', 'toothbrush']

    def __init__(self, nn, wh=(640, 640)):
        print("BaseYOLOv5Detector -> __init__")
        super().__init__(nn)
        self.nn = nn
        self._wh = wh  # 宽高比
        self.draw_box = True

    def init(self, model_path):
        print("BaseYOLOv5Detector -> init")
        # params = {"mean_values":[[0.0,0.0, 0.0]], "std_values":[[255.0, 255.0, 255.0]], "reorder_channel":'0 1 2'}
        self._nn.loadModel(model_path)

    # def init(self, model_path):
    #     print("YOLOv5 -> init")
    #     self._nn.loadModel(model_path, [[0.0,0.0, 0.0]], [[255.0, 255.0, 255.0]], '0 1 2')

    def detect(self, img):
        if isinstance(img, str):
            print("BaseYOLOv5Detector -> cv2.imread")
            img = cv2.imread(img)

        src_h, src_w = img.shape[:2]
        new_img, gain = self._preInference(img, self._wh)
        result = self._nn.inference(new_img)
        self._afterInference(result, gain, img, src_w, src_h)

    def _preInference(self, img, new_wh=(640, 640), color=(114, 114, 114)):
        print("BaseYOLOv5Detector -> _preInference")
        a = AutoScale(img, *new_wh)
        new_img = a.new_img
        h, w = new_img.shape[:2]
        new_img = cv2.copyMakeBorder(new_img, 0, new_wh[1] - h, 0, new_wh[0] - w, cv2.BORDER_CONSTANT, value=color)
        return new_img, (new_wh[0] / a.scale, new_wh[1] / a.scale)

    def _afterInference(self, result, gain, src_img, src_w, src_h):
        print("BaseYOLOv5Detector -> _afterInference gain: ", gain)
        # print("BaseYOLOv5Detector -> _afterInference result: ",
        #       len(result[0]), len(result[0][0]), len(result[0][0][0]), len(result[0][0][0][0]), len(result[0][0][0][0][0]))

        boxes, classes, scores = [], [], []
        for t in range(3):
            input0_data = sigmoid(result[t][0])
            input0_data = np.transpose(input0_data, (1, 2, 0, 3))
            grid_h, grid_w, channel_n, predict_n = input0_data.shape
            print("BaseYOLOv5Detector -> _afterInference grid_h / grid_w : ", grid_h, grid_w)
            anchors = [self._anchors[i] for i in self._masks[t]]
            box_confidence = input0_data[..., 4]
            box_confidence = np.expand_dims(box_confidence, axis=-1)
            box_class_probs = input0_data[..., 5:]
            box_xy = input0_data[..., :2]
            box_wh = input0_data[..., 2:4]
            col = np.tile(np.arange(0, grid_w), grid_h).reshape(-1, grid_w)
            row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_w)
            col = col.reshape((grid_h, grid_w, 1, 1)).repeat(3, axis=-2)
            row = row.reshape((grid_h, grid_w, 1, 1)).repeat(3, axis=-2)
            grid = np.concatenate((col, row), axis=-1)
            box_xy = box_xy * 2 - 0.5 + grid
            box_wh = (box_wh * 2) ** 2 * anchors
            box_xy /= (grid_w, grid_h)  # 计算原尺寸的中心
            box_wh /= self._wh  # 计算原尺寸的宽高
            box_xy -= (box_wh / 2.)  # 计算原尺寸的中心
            box = np.concatenate((box_xy, box_wh), axis=-1)
            res = self.__filter_boxes(box, box_confidence, box_class_probs, self._conf_thresh)
            boxes.append(res[0])
            classes.append(res[1])
            scores.append(res[2])
        boxes, classes, scores = np.concatenate(boxes), np.concatenate(classes), np.concatenate(scores)
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = self.__nms_boxes(b, s, self._iou_thresh)
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])
        if len(nboxes) < 1:
            return [], []
        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)
        label_list = []
        box_list = []
        for (x, y, w, h), score, cl in zip(boxes, scores, classes):
            print("gain[0:2]: ", gain[0:2])
            x *= gain[0]
            y *= gain[1]
            w *= gain[0]
            h *= gain[1]
            x1 = max(0, np.floor(x).astype(int))
            y1 = max(0, np.floor(y).astype(int))
            x2 = min(src_w, np.floor(x + w + 0.5).astype(int))
            y2 = min(src_h, np.floor(y + h + 0.5).astype(int))
            label_list.append(self._names[cl])
            box_list.append((x1, y1, x2, y2))
            if self.draw_box:
                self.__plotOneBox((x1, y1, x2, y2), src_img, label=self._names[cl])

        self._showResult(src_img)
        print("BaseYOLOv5Detector -> detect finished!")
        return label_list, box_list

    def _showResult(self, result):
        cv2.imwrite("../runs/rknn-detect/1.jpg", result)

    def __plotOneBox(self, x, img, color=None, label=None, line_thickness=None):
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def __filter_boxes(self, boxes, box_confidences, box_class_probs, conf_thres):
        box_scores = box_confidences * box_class_probs  # 条件概率， 在该cell存在物体的概率的基础上是某个类别的概率
        box_classes = np.argmax(box_scores, axis=-1)  # 找出概率最大的类别索引
        box_class_scores = np.max(box_scores, axis=-1)  # 最大类别对应的概率值
        pos = np.where(box_class_scores >= conf_thres)  # 找出概率大于阈值的item
        # pos = box_class_scores >= OBJ_THRESH  # 找出概率大于阈值的item
        boxes = boxes[pos]
        classes = box_classes[pos]
        scores = box_class_scores[pos]
        return boxes, classes, scores

    def __nms_boxes(self, boxes, scores, iou_thresh):
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep


"""
amicro indoor yolov5
"""


class AmicroIndoorYoloxDetector(BaseYOLOXDetector):
    """直接覆盖父类的属性，可以修改默认的锚点和iou值等"""
    _names = ["trashcan", "slippers", "wire", "socks", "carpet", "book",
              "feces", "curtain", "stool", "bed", "sofa", "closestool", "table",
              "cabinet", "ajardoor", "opendoor", "closedoor", "stairway"]
    nc = len(_names)
    no = nc + 5  # number of outputs per anchor
    nl = 3  # number of detection layers
    na = 3  # number of anchors

    def __init__(self, nn):
        super().__init__(nn, (640, 640))

    def _showResult(self, result):
        print("OriginYolov5Detector -> _showResult")
        cv2.imwrite("../runs/rknn-detect/1.jpg", result)

    def xywh2xyxy(self, x):
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def __filter_boxes(self, boxes, box_confidences, box_class_probs, conf_thres):
        box_scores = box_confidences * box_class_probs  # 条件概率， 在该cell存在物体的概率的基础上是某个类别的概率
        box_classes = np.argmax(box_scores, axis=-1)  # 找出概率最大的类别索引
        box_class_scores = np.max(box_scores, axis=-1)  # 最大类别对应的概率值
        pos = np.where(box_class_scores >= conf_thres)  # 找出概率大于阈值的item
        # pos = box_class_scores >= OBJ_THRESH  # 找出概率大于阈值的item
        boxes = boxes[pos]
        classes = box_classes[pos]
        scores = box_class_scores[pos]
        return boxes, classes, scores

    def draw(self, image, boxes, scores, classes):
        """Draw the boxes on the image.

        # Argument:
            image: original image.
            boxes: ndarray, boxes of objects.
            classes: ndarray, classes of objects.
            scores: ndarray, scores of objects.
            all_classes: all classes name.
        """
        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = box
            print('class: {}, score: {}'.format(self._names[cl], score))
            print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
            top = int(top)
            left = int(left)
            right = int(right)
            bottom = int(bottom)

            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(self._names[cl], score),
                        (top, left - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

    def detect_suit_yolo(self, img):
        result = self._nn.inference([img])
        if not result:
            print("Inference result is NULL")
            return []
        boxes, scores, classes = self._calMapPostprocess(result)
        if scores is None:
            return []
        output = np.concatenate((boxes, np.expand_dims(scores, axis=1), np.expand_dims(classes, axis=1)), axis=1)
        import cv2
        img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if boxes is not None:
            self.draw(img_1, boxes, scores, classes)
        cv2.imwrite("asd.jpg", img_1)
        return output

    def process(self, input, mask, anchors):

        anchors = [anchors[i] for i in mask]
        grid_h, grid_w = map(int, input.shape[0:2])

        box_confidence = sigmoid(input[..., 4])
        box_confidence = np.expand_dims(box_confidence, axis=-1)

        box_class_probs = sigmoid(input[..., 5:])

        box_xy = sigmoid(input[..., :2]) * 2 - 0.5

        col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
        col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)
        box_xy += grid
        box_xy *= int(self._wh[1] / grid_h)

        box_wh = pow(sigmoid(input[..., 2:4]) * 2, 2)
        box_wh = box_wh * anchors

        box = np.concatenate((box_xy, box_wh), axis=-1)

        return box, box_confidence, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes with object threshold.

        # Arguments
            boxes: ndarray, boxes of objects.
            box_confidences: ndarray, confidences of objects.
            box_class_probs: ndarray, class_probs of objects.

        # Returns
            boxes: ndarray, filtered boxes.
            classes: ndarray, classes for boxes.
            scores: ndarray, scores for boxes.
        """
        box_scores = box_confidences * box_class_probs
        box_classes = np.argmax(box_scores, axis=-1)
        box_class_scores = np.max(box_scores, axis=-1)
        pos = np.where(box_class_scores >= self._conf_thresh)

        boxes = boxes[pos]
        classes = box_classes[pos]
        scores = box_class_scores[pos]

        return boxes, classes, scores

    def nms_boxes(self, boxes, scores):
        """Suppress non-maximal boxes.

        # Arguments
            boxes: ndarray, boxes of objects.
            scores: ndarray, scores of objects.

        # Returns
            keep: ndarray, index of effective boxes.
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self._iou_thresh)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep

    def yolov5_post_process_full(self, input_data):
        boxes, classes, scores = [], [], []
        for input, mask in zip(input_data, self._masks):
            b, c, s = self.process(input, mask, self._anchors)
            b, c, s = self.filter_boxes(b, c, s)
            boxes.append(b)
            classes.append(c)
            scores.append(s)

        boxes = np.concatenate(boxes)
        boxes = self.xywh2xyxy(boxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)

        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]

            keep = self.nms_boxes(b, s)

            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return boxes, classes, scores

    def yolov5_post_process_for_cal_MAP(self, input_data):
        boxes, classes, scores = [], [], []
        for input, mask in zip(input_data, self._masks):
            b, c, s = self.process(input, mask, self._anchors)
            b, c, s = self.filter_boxes(b, c, s)
            boxes.append(b)
            classes.append(c)
            scores.append(s)

        boxes = np.concatenate(boxes)
        boxes = self.xywh2xyxy(boxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)

        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]

            keep = self.nms_boxes(b, s)

            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return boxes, classes, scores

    def _calMapPostprocess(self, result):
        print("BaseYOLOv5Detector -> _afterInference result: ", len(result[0]), len(
            result[0][0]), len(result[0][0][0]), len(result[0][0][0][0]))

        boxes, classes, scores = [], [], []

        bs, _, ny, nx = result[0].shape  # x(bs,255,80,80) to x(bs,3,80,80,85)
        input0_data = torch.tensor(result[0]).view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).cpu().numpy()
        bs, _, ny, nx = result[1].shape  # x(bs,255,40,40) to x(bs,3,40,40,85)
        input1_data = torch.tensor(result[1]).view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).cpu().numpy()
        bs, _, ny, nx = result[2].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        input2_data = torch.tensor(result[2]).view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).cpu().numpy()
        input0_data = input0_data.reshape(*input0_data.shape[1:])
        input1_data = input1_data.reshape(*input1_data.shape[1:])
        input2_data = input2_data.reshape(*input2_data.shape[1:])
        input_data = list()
        input_data.append(np.transpose(input0_data, (1, 2, 0, 3)))
        input_data.append(np.transpose(input1_data, (1, 2, 0, 3)))
        input_data.append(np.transpose(input2_data, (1, 2, 0, 3)))
        boxes, classes, scores = self.yolov5_post_process_full(input_data)

        print("AmicroYOLOv5Detector -> detect finished!")
        return boxes, scores, classes
