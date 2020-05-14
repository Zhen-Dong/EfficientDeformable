# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Zhen Dong
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging

import torch
import torch.nn as nn

# from .DCNv2.dcn_v2 import DCN
from pytorchcv.model_provider import get_model as ptcv_get_model
from thop import profile
from ..external.modules import dcn_deform_conv
import torch.utils.model_zoo as model_zoo

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

model_urls = {'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def channel_shuffle(x, G):
    N, C, H, W = x.size()
    x = x.view(N, G, C // G, H, W)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(N, C, H, W)
    return x


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class BaseNode(nn.Module):
    def __init__(self, inp, oup, stride, batch_norm, conv_kernel):
        super(BaseNode, self).__init__()
        self.stride = stride
        oup_inc = oup // 2

        if self.stride == 1:
            self.b2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                batch_norm(oup_inc, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                # dw
                conv_kernel(oup_inc, oup_inc, 3, 1, 1, groups=oup_inc, bias=False),
                batch_norm(oup_inc, momentum=BN_MOMENTUM),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                batch_norm(oup_inc, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )
        elif self.stride == 2:
            self.b1 = nn.Sequential(
                # dw
                conv_kernel(inp, inp, 3, 2, 1, groups=inp, bias=False),
                batch_norm(inp, momentum=BN_MOMENTUM),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                batch_norm(oup_inc, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )

            self.b2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                batch_norm(oup_inc, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                # dw
                conv_kernel(oup_inc, oup_inc, 3, 2, 1, groups=oup_inc, bias=False),
                batch_norm(oup_inc, momentum=BN_MOMENTUM),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                batch_norm(oup_inc, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        if 1 == self.stride:
            split = x.shape[1] // 2
            x1 = x[:, :split, :, :]
            x2 = x[:, split:, :, :]
            x2 = self.b2(x2)
        else:
            x1 = self.b1(x)
            x2 = self.b2(x)

        y = torch.cat((x1, x2), 1)
        y = channel_shuffle(y, 2)
        return y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseShuffleNetV2(nn.Module):

    def __init__(self, block, layers, heads, head_conv, w2=None, deform=False):
        self.w2 = w2
        self.deform = deform
        self.heads = heads
        self.deconv_with_bias = False

        super(PoseShuffleNetV2, self).__init__()

        if self.w2 == True:
            self.channels = [64, 256, 512, 1024, 2048]
            # self.channels = [24, 244, 488, 976, 2153]
        else:
            self.channels = [64, 128, 256, 512, 1024]
            # self.channels = [24, 116, 232, 464, 1024]
        self.layer0 = nn.Sequential(nn.Conv2d(3, self.channels[0], 3, 4, 1, bias=False),
            nn.BatchNorm2d(self.channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))

        stage_repeats = [3, 7, 3]
        for idx in range(len(stage_repeats)):
            if not self.deform:
                layers = [BaseNode(self.channels[idx], self.channels[idx + 1],
                               2, nn.BatchNorm2d, nn.Conv2d)]
            else:
                layers = [BaseNode(self.channels[idx], self.channels[idx + 1],
                               2, nn.BatchNorm2d, dcn_deform_conv.DeformConvWithOffsetScaleBoundPositive)]
            for _ in range(stage_repeats[idx]):
                if not self.deform:
                    layers.append(BaseNode(self.channels[idx],
                                       self.channels[idx + 1],
                                       1, nn.BatchNorm2d, nn.Conv2d))
                else:
                    layers.append(BaseNode(self.channels[idx],
                                           self.channels[idx + 1],
                                           1, nn.BatchNorm2d, dcn_deform_conv.DeformConvWithOffsetScaleBoundPositive))
            setattr(self, 'layer' + str(idx + 1), nn.Sequential(*layers))

        if not self.deform:
            self.layer4 = nn.Sequential(nn.Conv2d(self.channels[3], self.channels[4], 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(self.channels[4], momentum=BN_MOMENTUM),
                                    nn.ReLU(inplace=True))
        else:
            self.layer4 = nn.Sequential(dcn_deform_conv.DeformConvWithOffsetScaleBoundPositive(
                                        self.channels[3], self.channels[4], 1, 1, 0, bias=False),
                                        nn.BatchNorm2d(self.channels[4], momentum=BN_MOMENTUM),
                                        nn.ReLU(inplace=True))

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            num_layers=3,
            # num_filters=[512, 256, 128],
            num_filters=[256, 128, 128],
            # num_filters=[256, 128, 64],
            num_kernels=[3, 3, 3],
            # num_kernels=[4, 4, 4],
        )

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    # nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),

                    # BaseNode(64, head_conv, stride=1, conv_kernel=nn.Conv2d),

                    nn.Conv2d(128, head_conv, 1, 1, 0, bias=False),
                    # nn.Conv2d(64, head_conv, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(head_conv, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),
                    # dw
                    nn.Conv2d(head_conv, head_conv, 3, 1, 1, groups=head_conv, bias=False),
                    nn.BatchNorm2d(head_conv, momentum=BN_MOMENTUM),
                    # this part can be combined with next pw linear
                    # pw-linear
                    # nn.Conv2d(head_conv, head_conv, 1, 1, 0, bias=False),
                    # nn.BatchNorm2d(head_conv, momentum=BN_MOMENTUM),

                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                        kernel_size=1, stride=1,
                        padding=0, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(64, classes,
                  kernel_size=1, stride=1,
                  padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        if self.w2 == True:
            # deconv_planes = [2048, 512, 256]
            deconv_planes = [2048, 256, 128]
            # deconv_planes = [2153, 256, 128]
        else:
            deconv_planes = [1024, 256, 128]
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            # fc = nn.Conv2d(deconv_planes[i], planes, 3, 1, 1, bias=False)

            # hidden_state = 64
            # fc = nn.Sequential(
            #     # pw
            #     nn.Conv2d(deconv_planes[i], hidden_state, 1, 1, 0, bias=False),
            #     nn.BatchNorm2d(hidden_state, momentum=BN_MOMENTUM),
            #     nn.ReLU(inplace=True),
            #     # dw
            #     nn.Conv2d(hidden_state, hidden_state, 3, 1, 1, groups=hidden_state, bias=False),
            #     nn.BatchNorm2d(hidden_state, momentum=BN_MOMENTUM),
            #     # pw-linear
            #     nn.Conv2d(hidden_state, planes, 1, 1, 0, bias=False)
            # )

            # fc = dcn_deform_conv.ModulatedDeformConvPack(
            #         deconv_planes[i], planes, 3, 1, 1, bias=False)

            # fc = dcn_deform_conv.DeformConvWithOffsetScaleBoundPositive(
            #     deconv_planes[i], planes, 3, 1, 1, groups=planes, bias=False, BN_MOMENTUM=BN_MOMENTUM)

            # fc = dcn_deform_conv.DeformConvWithOffsetScaleBoundPositive(
            #     deconv_planes[i], planes, 3, 1, 1, groups=planes, bias=False,
            #                                        hidden_state=planes, BN_MOMENTUM=BN_MOMENTUM)

            fc = dcn_deform_conv.DeformConvWithOffsetScaleBoundPositive(
                deconv_planes[i], planes, 3, 1, 1, groups=planes, bias=False, hidden_state=128, BN_MOMENTUM=BN_MOMENTUM)

            # fc = nn.Conv2d(self.inplanes, planes,
            #         kernel_size=3, stride=1,
            #         padding=1, dilation=1, bias=False)

            # fill_fc_weights(fc)

            # up = nn.ConvTranspose2d(
            #         in_channels=planes,
            #         out_channels=planes,
            #         kernel_size=kernel,
            #         stride=2,
            #         padding=padding,
            #         output_padding=output_padding,
            #         bias=self.deconv_with_bias)
            # up = nn.ConvTranspose2d(
            #     in_channels=planes,
            #     out_channels=planes,
            #     kernel_size=kernel,
            #     stride=2,
            #     groups=planes,
            #     padding=padding,
            #     output_padding=output_padding,
            #     bias=self.deconv_with_bias)
            # When kernel size = 4, we have output_padding = 0, padding = 1,
            # so that H_out = 2 * H_in, W_out = 2 * W_in,
            # When kernel size = 3, we have output_padding = 1, padding = 1,
            # so that H_out = 2 * H_in, W_out = 2 * W_in,
            # it should be noted that the channel number is not changed

            up = nn.Upsample(size=None, scale_factor=2, mode='nearest', align_corners=None)

            # up = nn.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=None)

            # hidden_state_up = 2 * planes
            # up = nn.Sequential(
            #     # pw
            #     nn.Conv2d(planes, hidden_state_up, 1, 1, 0, bias=False),
            #     nn.BatchNorm2d(hidden_state_up, momentum=BN_MOMENTUM),
            #     nn.ReLU(inplace=True),
            #     # dw
            #     nn.ConvTranspose2d(
            #         in_channels=hidden_state_up,
            #         out_channels=hidden_state_up,
            #         kernel_size=kernel,
            #         stride=2,
            #         groups=hidden_state_up,
            #         padding=padding,
            #         output_padding=output_padding,
            #         bias=self.deconv_with_bias),
            #     nn.BatchNorm2d(hidden_state_up, momentum=BN_MOMENTUM),
            #     # pw-linear
            #     nn.Conv2d(hidden_state_up, planes, 1, 1, 0, bias=False)
            # )
            # fill_fc_weights(up)
            # for m in up.modules():
            #     if isinstance(m, nn.ConvTranspose2d):
            #         fill_up_weights(m)

            # hidden_state_up = 2 * planes
            # up = nn.Sequential(
            #     # pw
            #     nn.Conv2d(planes, hidden_state_up, 1, 1, 0, bias=False),
            #     nn.BatchNorm2d(hidden_state_up, momentum=BN_MOMENTUM),
            #     nn.ReLU(inplace=True),
            #     # dw
            #     nn.Upsample(size=None, scale_factor=2, mode='nearest', align_corners=None),
            #     # pw-linear
            #     nn.Conv2d(hidden_state_up, planes, 1, 1, 0, bias=False)
            # )
            # fill_fc_weights(up)

            # fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)

            # layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            # layers.append(nn.ReLU(inplace=True))

            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # print(x.is_cuda)
        x = self.deconv_layers(x)
        # print(x.shape)
        # the shape of x should be [16, 64, 128, 128]
        ret = {}
        # we set ctdet on, with self.heads = {'hm': 80, 'wh': 2, 'reg': 2}
        # training chunk size should be 16, so that the shape of ret[head]
        # should be [16, 80, 128, 128], [16, 2, 128, 128], [16, 2, 128, 128]
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
            # print(ret[head].shape)
        return [ret]

    def init_weights(self, num_layers):
        # pretrained model for ShuffleNetV2 BaseNodes
        # url = model_urls['shufflenetv2_x1.0']
        # pretrained_state_dict = model_zoo.load_url(url)
        # print('=> loading pretrained model {}'.format(url))
        #
        # modified_dict = {}
        # for key, value in pretrained_state_dict.items():
        #     modified_key = key.replace("stage2", "layer1") \
        #         .replace("stage3", "layer2").replace("stage4", "layer3") \
        #         .replace("branch", "b").replace("conv1", "layer0").replace("conv5", "layer4")
        #     modified_dict[modified_key] = value
        # print(self.load_state_dict(modified_dict, strict=False))

        if self.w2 == True:
            # self-defined pretrained model for Wide ShuffleNetV2 BaseNodes
            pretrained_state_dict = torch.load('/rscratch/'
                'zhendong/yaohuic/CenterNet/src/pretrain_logs/shufflenetv2_w2_512_trial1/model_best.pth.tar')['state_dict']
            print('=> loading self-defined wide ShuffleNetV2 pretrained model')
        else:
            # self-defined pretrained model for ShuffleNetV2 BaseNodes
            pretrained_state_dict = torch.load('/rscratch/'
                'zhendong/yaohuic/CenterNet/src/pretrain_logs/shufflenetv2_1_512_trial1/model_best.pth.tar')['state_dict']
            print('=> loading self-defined pretrained model')

        modified_dict = {}
        for key, value in pretrained_state_dict.items():
            modified_key = key.replace('module.', '').replace("stage2", "layer1") \
                .replace("stage3", "layer2").replace("stage4", "layer3") \
                .replace("branch", "b").replace("conv1", "layer0").replace("conv5", "layer4")
            modified_dict[modified_key] = value
        print(self.load_state_dict(modified_dict, strict=False))


        # pretrained PyTorchCV model for ShuffleNetV2 BaseNodes
        # if self.w2 == True:
        #     model_name = "shufflenetv2_w2"
        # else:
        #     model_name = "shufflenetv2_w1"
        # pretrained_state_dict = ptcv_get_model(model_name, pretrained=True).state_dict()
        # print('=> loading PyTorchCV pretrained model {}'.format(model_name))
        # modified_dict = {}
        # for key, value in pretrained_state_dict.items():
        #     modified_key = key.replace("features.stage1.", "layer1.") \
        #         .replace("features.stage2.", "layer2.").replace("features.stage3.", "layer3.") \
        #         .replace("unit1.", "0.").replace("unit2.", "1.").replace("unit3.", "2.") \
        #         .replace("unit4.", "3.").replace("unit5.", "4.").replace("unit6.", "5.") \
        #         .replace("unit7.", "6.").replace("unit8.", "7.")  \
        #         .replace("compress_layer0", "b2.0") \
        #         .replace("dw_conv2", "b2.3").replace("compress_bn1", "b2.1") \
        #         .replace("dw_bn2", "b2.4").replace("compress_conv1", "b2.0") \
        #         .replace("expand_conv3", "b2.5").replace("expand_bn3", "b2.6") \
        #         .replace("dw_conv4", "b1.0").replace("dw_bn4", "b1.1") \
        #         .replace("expand_conv5", "b1.2").replace("expand_bn5", "b1.3") \
        #         .replace("features.final_block.conv", "layer4.0").replace("features.final_block.bn", "layer4.1") \
        #         .replace("features.init_block.conv.conv", "layer0.0").replace("features.init_block.conv.bn", "layer0.1")
        #     modified_dict[modified_key] = value
        # print(self.load_state_dict(modified_dict, strict=False))

        # initialization for the ShuffleNet backbone
        # print('=> init stage0,1,2,3 weights from normal distribution')
        # for name, m in self.layer0.named_modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        # for name, m in self.layer1.named_modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        # for name, m in self.layer2.named_modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        # for name, m in self.layer3.named_modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)


def get_shufflenetv2_dcn(num_layers, heads, head_conv=64, deform_conv='ModulatedDeformConvPack'):
    # this is a placeholder, modules for ShuffleNetV2 are hardcoded in the model.
    block_class, layers = (BaseNode, [3, 7, 3, 1])

    model = PoseShuffleNetV2(block_class, layers, heads, head_conv=head_conv, w2=True, deform=False)
    model.init_weights(num_layers)

    # input = torch.randn(1, 3, 512, 512)
    # print(input.is_cuda)
    #
    # macs, params = profile(model, inputs=(input,))
    # print('MACs:', macs, 'Parameters:', params)

    return model
