import cv2
import numpy as np
import random
import torch
import sys
import os
from pathlib import Path
from torchvision.ops import nms, boxes
from rknn_lib.rknn_python_inference.detector_bridge import Detector


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
    _conf_thresh = 0.5
    _iou_thresh = 0.3
    _names = ["trashcan", "slippers", "wire", "socks", "carpet", "book",
              "feces", "curtain", "stool", "bed", "sofa", "closetool", "table",
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

    # def xywh2xyxy(self, x):
    #     # Convert [center_x, center_y, w, h] to left-top,right-bottom
    #     y = np.copy(x)
    #     y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    #     y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    #     y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    #     y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    #     return y

    # def __filter_boxes(self, boxes, box_confidences, box_class_probs, conf_thres):
    #     box_scores = box_confidences * box_class_probs  # 条件概率， 在该cell存在物体的概率的基础上是某个类别的概率
    #     box_classes = np.argmax(box_scores, axis=-1)  # 找出概率最大的类别索引
    #     box_class_scores = np.max(box_scores, axis=-1)  # 最大类别对应的概率值
    #     pos = np.where(box_class_scores >= conf_thres)  # 找出概率大于阈值的item
    #     # pos = box_class_scores >= OBJ_THRESH  # 找出概率大于阈值的item
    #     boxes = boxes[pos]
    #     classes = box_classes[pos]
    #     scores = box_class_scores[pos]
    #     return boxes, classes, scores

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
            x1, y1, x2, y2 = box

            print('class: {}, score: {}'.format(self._names[cl], score))
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print('box coordinate x1, y1, x2, y2: [{}, {}, {}, {}]'.format(x1, y1, x2, y2))

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(self._names[cl], score),
                        (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
        return image

    def letterbox(self, im, new_shape=(640, 640), color=(0, 0, 0)):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)

    def detect_suit_yolo(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)
        self.image_origin_shape = np.array(np.shape(img)[0:2])
        img_1, _, _ = self.letterbox(img, new_shape=self._wh)
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
        # new_img, _ = self._preInference(img, self._wh)
        cv2.imwrite("img_1.jpg", img_1)
        result = self._nn.inference([img_1])
        if not result:
            print("Inference result is NULL")
            return []
        boxes, scores, classes = self._calMapPostprocess(result)
        if scores is None:
            return []

        # output = np.array([[296, 464, 392, 498, 0.95, 6],
        #                    [80, 59, 473, 318, 0.95, 8]])
        # output = output[:, [1, 0, 3, 2, 4, 5]]
        # boxes = output[:, :4]
        # scores = output[:, 4]
        # classes = output[:, 5].astype(int)
        boxes = boxes[:, [1, 0, 3, 2]]
        output = np.concatenate((boxes, np.expand_dims(scores, axis=1), np.expand_dims(classes, axis=1)), axis=1)

        # img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if boxes is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_2 = self.draw(img, boxes, scores, classes)
        cv2.imwrite("img_2.jpg", img_2)
        return output

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

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        #-----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        #-----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            #-----------------------------------------------------------------#
            #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
            #-----------------------------------------------------------------#
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape

            box_yx = (box_yx - offset) * scale
            # box_yx = (box_yx) * scale
            box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2],
                               box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def decode_outputs(self, outputs):
        grids = []
        strides = []
        outputs = [torch.tensor(x) for x in outputs]
        hw = [x.shape[-2:] for x in outputs]
        #---------------------------------------------------#
        #   outputs输入前代表每个特征层的预测结果
        #   batch_size, 4 + 1 + num_classes, 80, 80 => batch_size, 4 + 1 + num_classes, 6400
        #   batch_size, 5 + num_classes, 40, 40
        #   batch_size, 5 + num_classes, 20, 20
        #   batch_size, 4 + 1 + num_classes, 6400 + 1600 + 400 -> batch_size, 4 + 1 + num_classes, 8400
        #   堆叠后为batch_size, 8400, 5 + num_classes
        #---------------------------------------------------#
        outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
        #---------------------------------------------------#
        #   获得每一个特征点属于每一个种类的概率
        #---------------------------------------------------#
        outputs[:, :, 4:] = torch.sigmoid(outputs[:, :, 4:])
        for h, w in hw:
            #---------------------------#
            #   根据特征层的高宽生成网格点
            #---------------------------#
            grid_y, grid_x = torch.meshgrid([torch.arange(h), torch.arange(w)])
            #---------------------------#
            #   1, 6400, 2
            #   1, 1600, 2
            #   1, 400, 2
            #---------------------------#
            grid = torch.stack((grid_x, grid_y), 2).view(1, -1, 2)
            shape = grid.shape[:2]

            grids.append(grid)
            strides.append(torch.full((shape[0], shape[1], 1), self._wh[0] / h))
        #---------------------------#
        #   将网格点堆叠到一起
        #   1, 6400, 2
        #   1, 1600, 2
        #   1, 400, 2
        #
        #   1, 8400, 2
        #---------------------------#
        grids = torch.cat(grids, dim=1).type(outputs.type())
        strides = torch.cat(strides, dim=1).type(outputs.type())
        #------------------------#
        #   根据网格点进行解码
        #------------------------#
        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        #-----------------#
        #   归一化
        #-----------------#
        outputs[..., [0, 2]] = outputs[..., [0, 2]] / self._wh[1]
        outputs[..., [1, 3]] = outputs[..., [1, 3]] / self._wh[0]
        return outputs

    def non_max_suppression(self, prediction, num_classes, input_shape,
                            image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
        #----------------------------------------------------------#
        #   将预测结果的格式转换成左上角右下角的格式。
        #   prediction  [batch_size, num_anchors, 85]
        #----------------------------------------------------------#
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        #----------------------------------------------------------#
        #   对输入图片进行循环，一般只会进行一次
        #----------------------------------------------------------#
        for i, image_pred in enumerate(prediction):
            #----------------------------------------------------------#
            #   对种类预测部分取max。
            #   class_conf  [num_anchors, 1]    种类置信度
            #   class_pred  [num_anchors, 1]    种类
            #----------------------------------------------------------#
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

            #----------------------------------------------------------#
            #   利用置信度进行第一轮筛选
            #----------------------------------------------------------#
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

            if not image_pred.size(0):
                continue
            #-------------------------------------------------------------------------#
            #   detections  [num_anchors, 7]
            #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
            #-------------------------------------------------------------------------#
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]

            nms_out_index = boxes.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thres,
            )

            output[i] = detections[nms_out_index]

            # #------------------------------------------#
            # #   获得预测结果中包含的所有种类
            # #------------------------------------------#
            # unique_labels = detections[:, -1].cpu().unique()

            # if prediction.is_cuda:
            #     unique_labels = unique_labels.cuda()
            #     detections = detections.cuda()

            # for c in unique_labels:
            #     #------------------------------------------#
            #     #   获得某一类得分筛选后全部的预测结果
            #     #------------------------------------------#
            #     detections_class = detections[detections[:, -1] == c]

            #     #------------------------------------------#
            #     #   使用官方自带的非极大抑制会速度更快一些！
            #     #------------------------------------------#
            #     keep = nms(
            #         detections_class[:, :4],
            #         detections_class[:, 4] * detections_class[:, 5],
            #         nms_thres
            #     )
            #     max_detections = detections_class[keep]

            #     # # 按照存在物体的置信度排序
            #     # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
            #     # detections_class = detections_class[conf_sort_index]
            #     # # 进行非极大抑制
            #     # max_detections = []
            #     # while detections_class.size(0):
            #     #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
            #     #     max_detections.append(detections_class[0].unsqueeze(0))
            #     #     if len(detections_class) == 1:
            #     #         break
            #     #     ious = bbox_iou(max_detections[-1], detections_class[1:])
            #     #     detections_class = detections_class[1:][ious < nms_thres]
            #     # # 堆叠
            #     # max_detections = torch.cat(max_detections).data

            #     # Add max detections to outputs
            #     output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

            if output[i] is not None:
                output[i] = output[i].cpu().numpy()
                box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4] = self.yolo_correct_boxes(
                    box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output

    def yolov5_post_process_full(self, input_data):
        boxes, classes, scores = [], [], []
        outputs = self.decode_outputs(input_data)
        results = self.non_max_suppression(outputs, self.nc,
                                           self._wh, self.image_origin_shape,
                                           True, conf_thres=self._conf_thresh,
                                           nms_thres=self._iou_thresh)
        classes = np.array(results[0][:, 6], dtype='int32')
        scores = results[0][:, 4] * results[0][:, 5]
        boxes = results[0][:, :4]

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

    # def yolov5_post_process_for_cal_MAP(self, input_data):
    #     boxes, classes, scores = [], [], []
    #     for input, mask in zip(input_data, self._masks):
    #         b, c, s = self.process(input, mask, self._anchors)
    #         b, c, s = self.filter_boxes(b, c, s)
    #         boxes.append(b)
    #         classes.append(c)
    #         scores.append(s)

    #     boxes = np.concatenate(boxes)
    #     boxes = self.xywh2xyxy(boxes)
    #     classes = np.concatenate(classes)
    #     scores = np.concatenate(scores)

    #     nboxes, nclasses, nscores = [], [], []
    #     for c in set(classes):
    #         inds = np.where(classes == c)
    #         b = boxes[inds]
    #         c = classes[inds]
    #         s = scores[inds]

    #         keep = self.nms_boxes(b, s)

    #         nboxes.append(b[keep])
    #         nclasses.append(c[keep])
    #         nscores.append(s[keep])

    #     if not nclasses and not nscores:
    #         return None, None, None

    #     boxes = np.concatenate(nboxes)
    #     classes = np.concatenate(nclasses)
    #     scores = np.concatenate(nscores)

    #     return boxes, classes, scores

    def _calMapPostprocess(self, result):
        print("BaseYOLOv5Detector -> _afterInference result: ", len(result[0]), len(
            result[0][0]), len(result[0][0][0]), len(result[0][0][0][0]))

        boxes, classes, scores = [], [], []

        input_data = list()
        input_data.append(result[0])
        input_data.append(result[1])
        input_data.append(result[2])
        boxes, classes, scores = self.yolov5_post_process_full(input_data)

        print("AmicroYOLOv5Detector -> detect finished!")
        return boxes, scores, classes
