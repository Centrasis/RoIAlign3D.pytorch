import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class CropAndResizeFunction(Function):

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, image, boxes, box_ind):
        pass

    def backward(ctx, *grad_outputs):
        pass

