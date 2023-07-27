# 2020.06.09-Changed for building GhostNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )
        # print('通道数量：', in_chs, out_chs, sep='\t')

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=512, width=1.0, dropout=0.2,embedding_dim=128):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                    se_ratio=se_ratio))
                input_channel = output_channel
            # print(output_channel)
            stages.append(nn.Sequential(*layers))

        # print(len(stages))
        # output_channel = _make_divisible(exp_size * width, 4)
        # stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        # input_channel = output_channel

        # self.blocks = nn.Sequential(*stages)

        self.blocks1 = nn.Sequential(
            stages[0],
            stages[1],
            stages[2],
            stages[3],
            stages[4],
        )
        self.blocks2 = nn.Sequential(
            stages[5],
            stages[6],
        )
        self.blocks3 = nn.Sequential(
            stages[7],
            stages[8],
        )

        # building last several layers
        output_channel = 256
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)
        self.embedding = nn.Linear(num_classes,embedding_dim)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        # x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x

"""

# 这是6m的那个模型
def ghostnet(**kwargs):

    # Constructs a GhostNet model

    cfgs = [
        # k, t, c, SE, s
        # stage1
        [[3, 16, 16, 0, 1]],
        # stage2
        [[3, 48, 24, 0, 2]],
        [[3, 72, 32, 0, 1]],
        # stage3
        [[5, 72, 32, 0, 2]],
        [[5, 120, 64, 0, 1]],
        # stage4
        [[3, 240, 64, 0, 2]],
        [[3, 200, 64, 0, 1],
         [3, 184, 128, 0, 1],
         [3, 184, 128, 0, 1],
         [3, 480, 128, 0, 1],
         [3, 672, 128, 0, 1]
         ],
        # stage5
        [[5, 672, 128, 0, 2]],
        [[5, 960, 256, 0, 1],
         [5, 960, 256, 0, 1],
         [5, 960, 256, 0, 1],
         [5, 960, 256, 0, 1]
         ]
    ]
    return GhostNet(cfgs, **kwargs)

"""
def ghostnet(**kwargs):

    #Constructs a GhostNet model

    cfgs = [
        # k, t, c, SE, s
        # stage1
        [[3, 16, 16, 0, 1]],
        # stage2
        [[3, 48, 24, 0, 2]],
        [[3, 72, 32, 0, 1]],
        # stage3
        [[5, 72, 32, 0.25, 2]],
        [[5, 120, 64, 0.25, 1]],
        # stage4
        [[3, 240, 64, 0, 2]],
        [[3, 200, 64, 0, 1],
         [3, 184, 128, 0, 1],
         [3, 184, 128, 0, 1],
         [3, 480, 128, 0.25, 1],
         [3, 672, 128, 0.25, 1]
         ],
        # stage5
        [[5, 672, 128, 0.25, 2]],
        [[5, 960, 256, 0, 1],
         [5, 960, 256, 0.25, 1],
         [5, 960, 256, 0, 1],
         [5, 960, 256, 0.25, 1]
         ]
    ]
    return GhostNet(cfgs, **kwargs)


# if __name__ == '__main__':
#     model = ghostnet()
#     model.eval()
#     print(model)
#     # input = torch.randn(32,3,320,256)
#     input = torch.randn(2,3,224,224)
#     y = model(input)
#     print(y)
#     print(y.shape)
#     print(y.shape)
# model = ghostnet()
# file_path = '/opt/ml/input/Retinaface_Ghost/weights/ghostnet_Final.pth'
# model.state_dict(torch.load(file_path))
# model.eval()

# x = torch.randn(11, 3, 224, 224)
# y = model(x)
# print(y.shape)

if __name__ == "__main__":
    ghostnet = ghostnet()
    file_path = '/opt/ml/input/Retinaface_Ghost/weights/ghostnet_Final.pth'
    ghostnet.state_dict(torch.load(file_path))
    # 레이어를 생성합니다.
    # layer = torch.nn.Linear(1000, 128)

# 레이어를 모델에 추가합니다.
    # ghostnet.add_module("layer", layer)

    
    # last_module = nn.Linear(1000, 32, bias=True)
    # ghostnet.add_module('last_module', last_module)
    # last_module.apply(ghostnet)#user_defined_initialize_function)
    
    
    # Create the additional layers
    # layer = nn.Linear(1000, 128)
    # last_module = nn.Linear(1000, 32, bias=True)

    # Add the layers to the ghostnet model
    # ghostnet.add_module("layer", layer)
    # ghostnet.add_module('last_module', last_module)

    # Apply initialization to the last_module
    # def user_defined_initialize_function(m):
        # Define your custom initialization logic here
        # pass

    # last_module.apply(user_defined_initialize_function)

    ghostnet.eval()
    
    # ghostnet.to(device)
    # x = torch.randn(11, 3, 224, 224).to(device)
    x = torch.randn(11,3,112,112) # scale no matter?
    y = ghostnet(x)
    print(y.shape)
    print(type(y))
    print(type(y[0]))