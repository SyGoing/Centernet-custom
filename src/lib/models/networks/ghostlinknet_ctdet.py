"""using ghostnet for backbone to face detection CVPR2020"""


"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch
"""
import torch
import torch.nn as nn
import math
import torch.onnx
import onnx
from onnx import optimizer

BN_MOMENTUM=0.01

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

def conv_1x1_bn(inp, oup):
    conv1x1=nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))
    for m in conv1x1.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    return conv1x1

def deconv_bn_relu(in_channels,out_channels,kernel_size,padding,output_padding,bias):
    deconv = nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            output_padding=output_padding,
            bias=bias),
        nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True)
    )
    for m in deconv.modules():
        if isinstance(m, nn.ConvTranspose2d):
            fill_up_weights(m)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    return deconv

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


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y


def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]   # Slice


class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # dw
            depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride==2 else nn.Sequential(),
            # Squeeze-and-Excite
            SELayer(hidden_dim) if use_se else nn.Sequential(),
            # pw-linear
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, 3, stride, relu=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Ghostlinknet(nn.Module):
    def __init__(self,heads,head_conv, width_mult=1. ,is_train=True):
        super(Ghostlinknet, self).__init__()
        self.cfgs=[
        # k,  t,    c, SE,s
        [3,  16,  16, 0, 1],   #1
        [3,  48,  24, 0, 2],   #2
        [3,  72,  24, 0, 1],   #3
        [5,  72,  40, 1, 2],   #4
        [5, 120,  40, 1, 1],  #5
        [3, 240,  80, 0, 2],  #6
        [3, 200,  80, 0, 1],  #7
        [3, 184,  80, 0, 1],  #8
        [3, 184,  80, 0, 1],  #9
        [3, 480, 112, 1, 1],  #10
        [3, 672, 112, 1, 1] , #11
        [5, 672, 160, 1, 2],  #12
        [5, 960, 160, 0, 1],  #13
        [5, 960, 160, 1, 1],  #14
        [5, 960, 160, 0, 1],  #15
        [5, 960, 160, 1, 1],  #16
    ]
        self.heads=heads
        self.head_conv=head_conv
        self.last_channel=24
        self.width_mult=width_mult
        self.is_train=is_train
        self.deconv_with_bias = False

        # building first layer
        output_channel = _make_divisible(16 * width_mult, 4)
        layers = [nn.Sequential(
            nn.Conv2d(3, output_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )]
        input_channel = output_channel

        # building inverted residual blocks
        block = GhostBottleneck
        for k, exp_size, c, use_se, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4)
            hidden_channel = _make_divisible(exp_size * width_mult, 4)
            layers.append(block(input_channel, hidden_channel, output_channel, k, s, use_se))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)

        #last layer
        self.backbone_lastlayer = conv_1x1_bn(input_channel, self.last_channel)

        #deconvolution
        self.ups = []
        for i in range(3):
            up = deconv_bn_relu(self.last_channel, self.last_channel, 2, 0, 0, self.deconv_with_bias)
            self.ups.append(up)
        self.ups = nn.Sequential(*self.ups)

        #matches for link add
        self.conv_dim_matchs = []
        self.conv_dim_matchs.append(conv_1x1_bn(112,self.last_channel))
        self.conv_dim_matchs.append(conv_1x1_bn(40, self.last_channel))
        self.conv_dim_matchs.append(conv_1x1_bn(24, self.last_channel))
        self.conv_dim_matchs = nn.Sequential(*self.conv_dim_matchs)

        for head in sorted(self.heads):
          num_output = self.heads[head]
          fc = nn.Conv2d(
              in_channels=self.last_channel,
              out_channels=num_output,
              kernel_size=1,
              stride=1,
              padding=0
          )
          self.__setattr__(head, fc)
          # if head_conv > 0:
          #   fc = nn.Sequential(
          #       nn.Conv2d(self.last_channel, head_conv,
          #         kernel_size=3, padding=1, bias=True),
          #       nn.ReLU(inplace=True),
          #       nn.Conv2d(head_conv, num_output,
          #         kernel_size=1, stride=1, padding=0))
          # else:
          #   fc = nn.Conv2d(
          #     in_channels=self.last_channel,
          #     out_channels=num_output,
          #     kernel_size=1,
          #     stride=1,
          #     padding=0
          # )



        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        xs = []

        for n in range(0, 4):
            x = self.features[n](x)
        xs.append(x)

        for n in range(4,6):
            x=self.features[n](x)
        xs.append(x)

        for n in range(6,12):
            x = self.features[n](x)
        xs.append(x)

        for n in range(12,17):
            x=self.features[n](x)

        x=self.backbone_lastlayer(x)

        for i in range(3):
            x=self.ups[i](x)
            x=x+self.conv_dim_matchs[i](xs[3-i-1])

        if self.is_train:
            ret = {}
            for head in self.heads:
                ret[head] = self.__getattr__(head)(x)
            return [ret]
        else:
            ret = []
            for head in self.heads:
                ret.append(self.__getattr__(head)(x))
            return ret


def get_ghostlink_net(num_layers, heads, head_conv,is_train=True):
  model = Ghostlinknet(heads, head_conv,1.0,is_train)
  return model

if __name__=='__main__':
    heads={'hm': 1, 'wh': 2, 'reg': 2}
    model=get_ghostlink_net(18,heads,64,False)
    model.eval()

    input = torch.randn(1, 3, 640, 640)
    y = model(input)

    torch.onnx.export(model,  # model being run
                      input,  # model input (or a tuple for multiple inputs)
                      "./test.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file# )
                      )

    print("finished !")




























