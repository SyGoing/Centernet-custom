

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

from opts import opts
from models.model import create_model, load_model

import torch
import torch.backends.cudnn as cudnn
import torch.onnx
import onnx
from onnx import optimizer

from PIL import Image
import torchvision.transforms as transforms


from torch.onnx import OperatorExportTypes
from collections import OrderedDict
from types import MethodType

if __name__ == '__main__':
    opt = opts().init()
    torch.manual_seed(0)
    cudnn.benchmark = True
    torch.set_default_tensor_type('torch.FloatTensor')

    # 设备定义
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # 网络模型加载
    model=create_model(opt.arch, opt.heads, opt.head_conv,is_train=False)
    model=load_model(model, opt.load_model)
    model.cuda()
    model.eval()


    # 图像预处理以及设备变化

    input = torch.randn(1, 3, 480, 640, requires_grad=True)
    input = input.to(device)

    out = model(input)

    torch.onnx.export(model,  # model being run
                      input,  # model input (or a tuple for multiple inputs)
                      "./onnxmodel/mv2relu.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file# )
                      )

print("hello finished")
