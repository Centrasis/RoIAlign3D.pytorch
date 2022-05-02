import math
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


if torch.cuda.is_available():
    import roi_align.crop_and_resize_gpu as crop_and_resize
else:
    import roi_align.crop_and_resize_cpu as crop_and_resize

class CropAndResizeFunction(Function):
    _fwd_func = crop_and_resize.forward2d
    _bwd_func = crop_and_resize.backward2d

    @classmethod
    def forward(cls, ctx, image, boxes, box_ind, crop_height, crop_width, extrapolation_value=0):
        ctx.crop_height = crop_height
        ctx.crop_width = crop_width
        ctx.extrapolation_value = extrapolation_value
        crops = torch.zeros_like(image)

        cls._fwd_func(
            image, boxes, box_ind,
            ctx.extrapolation_value, ctx.crop_height, ctx.crop_width, crops
            )

        # save for backward
        ctx.im_size = image.size()
        ctx.save_for_backward(boxes, box_ind)

        return crops

    @classmethod
    def backward(cls,ctx, grad_outputs):
        boxes, box_ind = ctx.saved_tensors

        grad_outputs = grad_outputs.contiguous()
        grad_image = torch.zeros_like(grad_outputs).resize_(*ctx.im_size)

        cls._bwd_func(
            grad_outputs, boxes, box_ind, grad_image
        )

        return grad_image, None, None, None, None, None


class CropAndResizeFunction3d(CropAndResizeFunction):
    _fwd_func = crop_and_resize.forward3d
    _bwd_func = crop_and_resize.backward3d


class CropAndResize(nn.Module):
    """
    Crop and resize ported from tensorflow
    See more details on https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize
    """

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        super(CropAndResize, self).__init__()

        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, image, boxes, box_ind):
        return CropAndResizeFunction.apply(image, boxes, box_ind, self.crop_height, self.crop_width, self.extrapolation_value)


class CropAndResize3d(CropAndResize):
    def __init__(self, crop_height: int, crop_width: int, crop_depth: int, extrapolation_value: int = 0):
        super().__init__(crop_height, crop_width, extrapolation_value)
        self.crop_depth = crop_depth

    def forward(self, image, boxes, box_ind):
        return CropAndResizeFunction3d.apply(image, boxes, box_ind, self.crop_height, self.crop_width, self.crop_depth, self.extrapolation_value)
