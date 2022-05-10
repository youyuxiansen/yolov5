import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
import torch
from rknn.api import RKNN


ONNX_MODEL = '/home/felix/project/yolov5/runs/train/indoor/exp6/weights/best.onnx'
RKNN_MODEL = '/home/felix/project/yolov5/runs/train/indoor/exp6/weights/best.rknn'
IMG_PATH = '/home/felix/project/yolov5/rknn_lib/rknn_python_inference/indoor1_000021.jpg'
DATASET = './dataset.txt'

OBJ_THRESH = 0.5
NMS_THRESH = 0.6
IMG_SIZE = 640

CLASSES = ('feces', 'slippers', 'closetool', 'cabinet', 'curtain',
           'trashcan', 'stool', 'table', 'bed', 'book', 'socks',
           'sofa', 'carpet', 'wire', 'Door', 'Charger')

masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
anchors = [[61, 40], [81, 106], [188, 61], [137, 155], [
    112, 316], [201, 228], [396, 210], [251, 368], [441, 362]]

nc = len(CLASSES)
no = nc + 5  # number of outputs per anchor
nl = 3  # number of detection layers
na = 3  # number of anchors


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def process(input, mask, anchors):

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
    box_xy *= int(IMG_SIZE / grid_h)

    box_wh = pow(sigmoid(input[..., 2:4]) * 2, 2)
    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs


def filter_boxes(boxes, box_confidences, box_class_probs):
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
    pos = np.where(box_class_scores >= OBJ_THRESH)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
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
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov5_post_process_simple(prediction):
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > OBJ_THRESH
    valid_object = prediction[xc]
    valid_object[:, 5:] *= valid_object[:, 4:5]

    boxes = xywh2xyxy(valid_object[:, :4])
    best_score_class = np.max(valid_object[:, 5:], axis=-1)
    box_classes = np.argmax(valid_object[:, 5:], axis=-1)

    nboxes, nclasses, nscores = [], [], []
    for c in set(box_classes):
        inds = np.where(box_classes == c)
        b = boxes[inds]
        c = box_classes[inds]
        s = best_score_class[inds]

        keep = nms_boxes(b, s)
        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def yolov5_post_process_full(input_data):

    boxes, classes, scores = [], [], []
    for input, mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw(image, boxes, scores, classes):
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
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
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


def calMapPostprocess(result):
    print("BaseYOLOv5Detector -> _afterInference result: ", len(result[0]), len(
        result[0][0]), len(result[0][0][0]), len(result[0][0][0][0]))

    boxes, classes, scores = [], [], []

    bs, _, ny, nx = result[0].shape  # x(bs,255,80,80) to x(bs,3,80,80,85)
    input0_data = torch.tensor(result[0]).view(bs, na, no, ny, nx).permute(0, 1, 3, 4, 2).cpu().numpy()
    bs, _, ny, nx = result[1].shape  # x(bs,255,40,40) to x(bs,3,40,40,85)
    input1_data = torch.tensor(result[1]).view(bs, na, no, ny, nx).permute(0, 1, 3, 4, 2).cpu().numpy()
    bs, _, ny, nx = result[2].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
    input2_data = torch.tensor(result[2]).view(bs, na, no, ny, nx).permute(0, 1, 3, 4, 2).cpu().numpy()
    input0_data = input0_data.reshape(*input0_data.shape[1:])
    input1_data = input1_data.reshape(*input1_data.shape[1:])
    input2_data = input2_data.reshape(*input2_data.shape[1:])
    input_data = list()
    input_data.append(np.transpose(input0_data, (1, 2, 0, 3)))
    input_data.append(np.transpose(input1_data, (1, 2, 0, 3)))
    input_data.append(np.transpose(input2_data, (1, 2, 0, 3)))
    boxes, classes, scores = yolov5_post_process_full(input_data)

    print("AmicroYOLOv5Detector -> detect finished!")
    return boxes, scores, classes


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()

    if not os.path.exists(ONNX_MODEL):
        print('model not exist')
        exit(-1)

    # pre-process config
    print('--> Config model')
    rknn.config(reorder_channel='0 1 2',
                mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                optimization_level=3, target_platform='RV1126')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load yolov5 failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build yolov5 failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    # print('--> Export RKNN model')
    # ret = rknn.export_rknn(RKNN_MODEL)
    # if ret != 0:
    #     print('Export yolov5rknn failed!')
    #     exit(ret)
    # print('done')

    # init runtime environment
    print('--> Init runtime environment')
    # ret = rknn.init_runtime()
    ret = rknn.init_runtime('RV1126', device_id="f3895196d12833e0")
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread(IMG_PATH)
    img, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])

    # simple post process
    # boxes, classes, scores = yolov5_post_process_simple(outputs[0])

    # img_0 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # if boxes is not None:
    #     draw(img_0, boxes, scores, classes)
    # cv2.imshow("direct result", img_0)

    if not outputs:
        print("Inference result is NULL")
        exit

    boxes, scores, classes = calMapPostprocess(outputs)
    if scores is None:
        exit
    output = np.concatenate((boxes, np.expand_dims(scores, axis=1), np.expand_dims(classes, axis=1)), axis=1)
    import cv2
    img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if boxes is not None:
        draw(img_1, boxes, scores, classes)
    cv2.imwrite("asd.jpg", img_1)
