import datetime
from pathlib import Path
import sys
from rknn.api import RKNN
from rknn_lib.rknn_python_inference.detector_bridge import NN

# 接收onnx类型的RKNN


class RknnImpl(NN):
    def __init__(self, **params):
        print("RknnImpl -> __init__")
        self.__params = params

    def loadModel(self, model_path):
        print("RknnImpl -> loadModel")
        rknn = RKNN()
        self.rknn = rknn
        # pre-process config
        print('--> Config model')
        # rknn.config(mean_values=[[0.0,0.0, 0.0]], std_values=[[255.0, 255.0, 255.0]], reorder_channel='0 1 2')

        # rknn.config(mean_values=[[0.] * 12], std_values=[[255.0] * 12], reorder_channel='')
        rknn.config(mean_values=self.__params["mean_values"],
                    std_values=self.__params["std_values"],
                    reorder_channel=self.__params["reorder_channel"],
                    output_optimize=1, target_platform='RV1126')

        print('--> Loading model', rknn.__dict__)
        if Path(model_path).suffix == ".onnx":
            # Load ONNX model
            ret = rknn.load_onnx(model_path)
            nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            onnx_filename_with_time = str(Path(model_path).stem) + '_' + str(nowTime)
            rknn_model_path = Path(model_path).parent / Path(onnx_filename_with_time).with_suffix('.rknn')
            # assert not rknn_model_path.exists(), f'failed to export rknn file cause it exists: {rknn_model_path}'

            # Build model
            print('--> Building model')
            # target_platform default rk1808.
            # ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
            ret = rknn.build(do_quantization=False)

            # ret = rknn.export_rknn(rknn_model_path)
        else:
            ret = rknn.load_rknn(model_path, load_model_in_npu=False)

        # 初始化RKNN运行环境
        print('--> Init runtime environment')
        # ret = rknn.init_runtime(host='rk3399pro')
        ret = rknn.init_runtime('RV1126')
        return rknn

    def inference(self, img):
        return self.rknn.inference(img)
