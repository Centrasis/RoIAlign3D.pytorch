import torch
from torch import Tensor, nn
from .crop_and_resize import CropAndResizeFunction, CropAndResizeFunction3d


class RoIAlign(nn.Module):

    def __init__(self, crop_height, crop_width, extrapolation_value=0, transform_fpcoor=True):
        super(RoIAlign, self).__init__()

        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value
        self.transform_fpcoor = transform_fpcoor

    def forward(self, featuremap, boxes, box_ind):
        """
        RoIAlign based on crop_and_resize.
        See more details on https://github.com/ppwwyyxx/tensorpack/blob/6d5ba6a970710eaaa14b89d24aace179eb8ee1af/examples/FasterRCNN/model.py#L301
        :param featuremap: NxCxHxW
        :param boxes: Mx4 float box with (x1, y1, x2, y2) **without normalization**
        :param box_ind: M
        :return: MxCxoHxoW
        """
        x1, y1, x2, y2 = torch.split(boxes, 1, dim=1)
        image_height, image_width = featuremap.size()[2:4]

        if self.transform_fpcoor:
            spacing_w = (x2 - x1) / float(self.crop_width)
            spacing_h = (y2 - y1) / float(self.crop_height)

            nx0 = (x1 + spacing_w / 2 - 0.5) / float(image_width - 1)
            ny0 = (y1 + spacing_h / 2 - 0.5) / float(image_height - 1)
            nw = spacing_w * float(self.crop_width - 1) / float(image_width - 1)
            nh = spacing_h * float(self.crop_height - 1) / float(image_height - 1)

            boxes = torch.cat((ny0, nx0, ny0 + nh, nx0 + nw), 1)
        else:
            x1 = x1 / float(image_width - 1)
            x2 = x2 / float(image_width - 1)
            y1 = y1 / float(image_height - 1)
            y2 = y2 / float(image_height - 1)
            boxes = torch.cat((y1, x1, y2, x2), 1)

        boxes = boxes.detach().contiguous()
        box_ind = box_ind.detach()
        return CropAndResizeFunction.apply(featuremap, boxes, box_ind, self.crop_height, self.crop_width, self.extrapolation_value)


class RoIAlign3d(RoIAlign):
    def __init__(self, crop_height, crop_width, crop_depth, extrapolation_value=0, transform_fpcoor=True):
        super().__init__(crop_height, crop_width, extrapolation_value, transform_fpcoor)
        self.crop_depth = crop_depth

    def forward(self, featuremap: Tensor, boxes: Tensor, box_ind: Tensor) -> Tensor:
        x1, y1, z1, x2, y2, z2 = torch.split(boxes, 1, dim=1)
        image_height, image_width, image_depth = featuremap.size()[2:5]

        if self.transform_fpcoor:
            spacing_w = (x2 - x1) / float(self.crop_width)
            spacing_h = (y2 - y1) / float(self.crop_height)
            spacing_d = (z2 - z1) / float(self.crop_depth)

            nx0 = (x1 + spacing_w / 2 - 0.5) / float(image_width - 1)
            ny0 = (y1 + spacing_h / 2 - 0.5) / float(image_height - 1)
            nz0 = (z1 + spacing_d / 2 - 0.5) / float(image_depth - 1)
            nw = spacing_w * float(self.crop_width - 1) / float(image_width - 1)
            nh = spacing_h * float(self.crop_height - 1) / float(image_height - 1)
            nz = spacing_d * float(self.crop_depth - 1) / float(image_depth - 1)

            boxes = torch.cat((nz0, ny0, nx0, nz0 + nz, ny0 + nh, nx0 + nw), 1)
        else:
            x1 = x1 / float(image_width - 1)
            x2 = x2 / float(image_width - 1)
            y1 = y1 / float(image_height - 1)
            y2 = y2 / float(image_height - 1)
            z1 = z1 / float(image_depth - 1)
            z2 = z2 / float(image_depth - 1)
            boxes = torch.cat((z1, y1, x1, z2, y2, x2), 1)
        return CropAndResizeFunction3d.apply(featuremap, boxes, box_ind, self.crop_height, self.crop_width, self.crop_depth, self.extrapolation_value)
