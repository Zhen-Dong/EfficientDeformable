import numpy as np
import math
import os
import string
import torch
from torch import nn
import torch.nn.functional as F

import torch.utils.model_zoo as model_zoo
from ..external.modules import dcn_deform_conv

model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
}

def channel_shuffle(x, G):
    N, C, H, W = x.size()
    x = x.view(N, G, C // G, H, W)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(N, C, H, W)
    return x


class BaseNode(nn.Module):
    def __init__(self, inp, oup, stride, batch_norm, conv_kernel):
        super(BaseNode, self).__init__()
        self.stride = stride
        oup_inc = oup // 2
        
        if self.stride == 1:
            self.b2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                batch_norm(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                conv_kernel(oup_inc, oup_inc, 3, 1, 1, groups=oup_inc, bias=False),
                batch_norm(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                batch_norm(oup_inc),
                nn.ReLU(inplace=True),
            )
        elif self.stride == 2:
            self.b1 = nn.Sequential(
                # dw
                conv_kernel(inp, inp, 3, 2, 1, groups=inp, bias=False),
                batch_norm(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                batch_norm(oup_inc),
                nn.ReLU(inplace=True),
            )
    
            self.b2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                batch_norm(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                conv_kernel(oup_inc, oup_inc, 3, 2, 1, groups=oup_inc, bias=False),
                batch_norm(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                batch_norm(oup_inc),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        if 1 == self.stride:
            split = x.shape[1]//2
            x1 = x[:, :split, :, :]
            x2 = x[:, split:, :, :]
            x2 = self.b2(x2)
        else:
            x1 = self.b1(x)
            x2 = self.b2(x)
        
        y = torch.cat((x1, x2), 1)
        y = channel_shuffle(y, 2)
        return y


class ModifiedBaseNode(nn.Module):
    def __init__(self, inp, oup, stride, batch_norm, conv_kernel):
        super(ModifiedBaseNode, self).__init__()
        self.stride = stride
        oup_inc = oup // 2

        if self.stride == 1:
            self.b2 = nn.Sequential(
                # pw
                nn.Conv2d(inp // 2, oup_inc, 1, 1, 0, bias=False),
                batch_norm(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                conv_kernel(oup_inc, oup_inc, 3, 1, 1, groups=oup_inc, bias=False),
                batch_norm(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                batch_norm(oup_inc),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        split = x.shape[1] // 2
        x1 = x[:, :split, :, :]
        x2 = x[:, split:, :, :]
        x2 = self.b2(x2)

        y = torch.cat((x1, x2), 1)
        y = channel_shuffle(y, 2)
        return y


class ShuffleNetV2(nn.Module):
    def __init__(self, batch_norm=nn.BatchNorm2d, pretrained=True):
        super(ShuffleNetV2, self).__init__()
        self.channels = [24, 116, 232, 464]
        # self.channels = [64, 128, 256, 512]

        self.layer0 = nn.Sequential(
                    nn.Conv2d(3, self.channels[0],
                              3, 4, 1, bias=False),
                    batch_norm(self.channels[0]),
                    nn.ReLU(inplace=True)
                )    
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # stage_repeats = [3, 5, 3]
        stage_repeats = [3, 7, 3]
        for idx in range(len(stage_repeats)):
            layers = [BaseNode(self.channels[idx], 
                               self.channels[idx+1], 
                               2, batch_norm, nn.Conv2d)]
            for _ in range(stage_repeats[idx]):
                layers.append(BaseNode(self.channels[idx], 
                                       self.channels[idx+1], 
                                       1, batch_norm, nn.Conv2d))
            setattr(self, 'layer' + str(idx+1), nn.Sequential(*layers))

        if pretrained:
            url = model_urls['shufflenetv2_x1.0']
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            modified_dict = {}
            for key, value in pretrained_state_dict.items():
                modified_key = key.replace("stage2", "layer1")\
                    .replace("stage3", "layer2").replace("stage4", "layer3")\
                    .replace("branch", "b").replace("conv1", "layer0")
                modified_dict[modified_key] = value
            print(self.load_state_dict(modified_dict, strict=False))

            # self.load_state_dict(torch.load('../../../models/shufflenet_v2_64.pth'), strict=False)
        # if pretrained:
        #     self.load_state_dict(torch.load(os.path.dirname(os.path.abspath(__file__)) + '/shufflenet_v2.pth'), strict=True)

    def forward(self, x):
        y = []
        x = self.layer0(x)
        # x = self.maxpool(x)
        y.append(x)
        x = self.layer1(x)
        y.append(x)
        x = self.layer2(x)
        y.append(x)
        x = self.layer3(x)
        y.append(x)
        return y

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.01)
            # nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            # nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()


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


class HeadNode(nn.Module):
    def __init__(self, inp, oup, batch_norm, conv_kernel):
        super(HeadNode, self).__init__()
        oup_inc = oup // 2
        # self.a = oup_inc
        self.b1 = nn.Sequential(
            # dw
            conv_kernel(inp, inp, 3, 1, 1, groups=inp, bias=False),
            batch_norm(inp),
            # pw-linear
            nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
            batch_norm(oup_inc),
            nn.ReLU(inplace=True),
        )        

        self.b2 = nn.Sequential(
            # pw
            nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
            batch_norm(oup_inc),
            nn.ReLU(inplace=True),
            # dw
            conv_kernel(oup_inc, oup_inc, 3, 1, 1, groups=oup_inc, bias=False),
            batch_norm(oup_inc),
            # pw-linear
            nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
            batch_norm(oup_inc),
            nn.ReLU(inplace=True),
        )      

    def forward(self, x):
        x1 = self.b1(x)
        x2 = self.b2(x)
        # print("oup_inc = ", self.a)
        x = torch.cat((x1, x2), 1)
        x = channel_shuffle(x, 2)
        return x

    # def forward(self, x1, x2=None):
    #     if x2 is None:
    #         x2 = self.b2(x1)
    #         x1 = self.b1(x1)
    #     else:
    #         x2 = self.b2(x2)
    #         x1 = self.b1(x1)

    #     x = torch.cat((x1, x2), 1)
    #     x = channel_shuffle(x, 2)
        # return x

# class HeadNode(nn.Module):
#     def __init__(self, chi, cho, batch_norm, conv_kernel):
#         super(HeadNode, self).__init__()
#         self.actf = nn.Sequential(
#             batch_norm(cho),
#             nn.ReLU(inplace=True)
#         )
#         self.conv = conv_kernel(chi, cho)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.actf(x)
#         return x


class IDAUp(nn.Module):

    def __init__(self, batch_norm, deform_conv, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(len(channels)):
            c = channels[i]
            f = int(up_f[i])

            proj = HeadNode(c, o, batch_norm, deform_conv)
            node = HeadNode(o, o, batch_norm, deform_conv)

            if c == o:
                proj = nn.Identity()

            if f == 1:
                up = nn.Identity()
            else:
                # up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                #                         padding=f // 2, output_padding=0,
                #                         groups=o, bias=False)
                # fill_up_weights(up)
                # up = Interpolate(f, 'nearest')
                up = nn.Upsample(scale_factor=f, mode='nearest')

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            if i > 0:
                setattr(self, 'node_' + str(i), node)
                 
    def forward(self, layers, startp, endp):
        layers[startp] = self.up_0(self.proj_0(layers[startp]))
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node((layers[i] + layers[i - 1]) / 2)
            # layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):

    def __init__(self, batch_norm, deform_conv, startp, channels, scales, in_channels):
        super(DLAUp, self).__init__()
        if in_channels is None:
            in_channels = list(channels)
        in_channels = np.array(in_channels, dtype=int)
        self.startp = startp
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(batch_norm, deform_conv, channels[j],
                          in_channels[j:], scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j:] = channels[j]

    def forward(self, layers):
        out = []
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class SFLNet(nn.Module):
    def __init__(self, heads, head_conv=0, zoom_factor=4,
                        deform_conv='DeformConvPack', final_kernel=1,
                        batch_norm=nn.BatchNorm2d, pretrained=True, 
                        channels=[64, 64, 64, 64]):
        super(SFLNet, self).__init__()
        assert zoom_factor in [4, 8]
        self.zoom_factor = zoom_factor
        self.first_level = int(np.log2(zoom_factor)) - 1
        self.last_level = len(channels) - 1
        self.base = ShuffleNetV2(batch_norm, pretrained)
        deform_conv_func = getattr(dcn_deform_conv, deform_conv)

        self.dla_up = DLAUp(batch_norm, deform_conv_func, self.first_level, 
                            channels[self.first_level:], 
                            [2 ** i for i in range(len(channels) - self.first_level)], 
                            self.base.channels[self.first_level:])
        
        out_dim = channels[self.first_level]        
        self.ida_up = IDAUp(batch_norm, deform_conv_func, out_dim, 
                        channels[self.first_level:self.last_level][::-1], 
                        [2 ** i for i in range(self.last_level - self.first_level)][::-1])

        # self.fc = nn.Sequential(
        #     nn.Conv2d(out_dim, classes, 
        #               kernel_size=1, stride=1, 
        #               padding=0, bias=True),
        #     # Interpolate(zoom_factor, 'nearest')
        #     # Interpolate(zoom_factor, 'bilinear')
        #     nn.Upsample(scale_factor=zoom_factor, mode='nearest')
        #     # nn.Upsample(scale_factor=zoom_factor, mode='bilinear')
        # )

        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    # nn.Conv2d(out_dim, head_conv, kernel_size=3, padding=1, bias=True),

                    # BaseNode(out_dim, head_conv, 1, nn.BatchNorm2d, nn.Conv2d),
                    # ModifiedBaseNode(out_dim, head_conv, 1, nn.BatchNorm2d, nn.Conv2d),
                    nn.Conv2d(out_dim, head_conv, 1, 1, 0, bias=False),
                    batch_norm(head_conv),
                    nn.ReLU(inplace=True),
                    # dw
                    nn.Conv2d(head_conv, head_conv, 3, 1, 1, groups=head_conv, bias=False),
                    batch_norm(head_conv),
                    # pw-linear
                    nn.Conv2d(head_conv, head_conv, 1, 1, 0, bias=False),
                    batch_norm(head_conv),

                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=final_kernel, stride=1,
                              padding=final_kernel // 2, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(out_dim, classes,
                               kernel_size=final_kernel, stride=1,
                               padding=final_kernel // 2, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)


    def forward(self, images):
        base_feats = self.base(images)
        pyramid_feats = self.dla_up(base_feats)

        final_feats = []
        for idx in range(self.last_level - self.first_level):
            final_feats.insert(0, pyramid_feats[idx].clone())
        self.ida_up(final_feats, 0, len(final_feats))

        up_feat = F.interpolate(final_feats[-1], scale_factor=2, mode='nearest')

        outputs = {}
        for head in self.heads:
            outputs[head] = self.__getattr__(head)(up_feat)
        return [outputs]

def get_sfl_dcn(num_layers, heads, head_conv=0, down_ratio=4, deform_conv='DeformConvPack'):
    model = SFLNet(heads, head_conv, down_ratio, deform_conv)
    return model