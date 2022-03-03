import cv2
import torch
import numpy as np
from rknn.api import RKNN
PT_PATH = './resnet18.pt'
RKNN_PATH = './resnet18.rknn'
MEANS = [123.675, 116.28, 103.53]
STDS = [58.395, 58.395, 58.395]


def inference_with_rknn(target, device_id, inputs):
    # 创建 RKNN 对象
    rknn = RKNN()
    # 从当前目录加载 RKNN 模型 resnet_18
    ret = rknn.load_rknn(path=RKNN_PATH)
    if ret != 0:
        print('Load Pytorch model failed!')
        exit(ret)
    # 初始化运行时环境
    ret = rknn.init_runtime(target=target, device_id=device_id)
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    # 调用 inference 接口进行推理
    outputs = rknn.inference(inputs=inputs)
    # 释放 RKNN 对象
    rknn.release()
    return outputs


def inference_with_torch(img):
    # 输入图像预处理
    img = (img - MEANS) / STDS
    # OpenCV 读取的图像是按 HWC 排列的,Torch 要求 NCHW,需要做下转换
    img = img.reshape((1, 224, 224, 3))
    img = img.transpose((0, 3, 1, 2))
    img = img.astype(np.float32)
    torch_inputs = [torch.from_numpy(img)]

    # 加载 PyTorch 模型
    net = torch.load(PT_PATH)
    # 调用 forward 方法推理
    outputs = net.forward(*torch_inputs)
    return outputs


def compute_cos_dis(x, y):
    cos_dist = (x * y) / (np.linalg.norm(x) * (np.linalg.norm(y)))
    return cos_dist.sum()


if __name__ == '__main__':
    # 读取测试图像
    img = cv2.imread('./space_shuttle_224.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 使用 RKNN Toolkit 推理接口在 RK1808 开发板上推理
    rknn_outs = inference_with_rknn(target='rk1109', device_id='1808', inputs=[img])

    # 使用 torch 进行推理
    torch_outs = inference_with_torch(img=img)
    # 计算并打印两个结果的余弦相似度
    cos_dis = compute_cos_dis(rknn_outs[0], torch_outs[0].cpu().detach().numpy())
    print("Cosine distance of RKNN output and Torch output: {}".format(cos_dis))
