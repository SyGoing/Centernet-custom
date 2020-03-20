import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet

from torchvision import transforms


def conv_bn(inp, oup, stride):

    conv_3x3=nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )
    for m in conv_3x3.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    return conv_3x3


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size, 1, padding, groups=groups, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(out_planes),)

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


class Encoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(Encoder, self).__init__()
        self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias)
        self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        return x


class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
        # self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
        #                         nn.BatchNorm2d(in_planes//4),
        #                         nn.ReLU(inplace=True),)
        # self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
        #                         nn.BatchNorm2d(in_planes//4),
        #                         nn.ReLU(inplace=True),)
        # self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
        #                         nn.BatchNorm2d(out_planes),
        #                         nn.ReLU(inplace=True),)

        self.decode=nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 4, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(in_planes // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_planes // 4, in_planes // 4, kernel_size, stride, padding, output_padding, bias=bias),
            nn.BatchNorm2d(in_planes // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // 4, out_planes, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        # 等待调试
        for m in self.decode.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                # print('=> init {}.weight as 1'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                #nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.tp_conv(x)
        # x = self.conv2(x)
        x=self.decode(x)

        return x


class ReLinkNet(nn.Module):
    """
    Generate Model Architecture
    """

    def __init__(self,num_layers, heads, head_conv,is_train):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(ReLinkNet, self).__init__()

        self.heads=heads
        self.head_conv=head_conv
        self.is_train=is_train

        base = resnet.resnet18(pretrained=False)
        if num_layers==18:
            base = resnet.resnet18(pretrained=True)
        elif num_layers==34:
            base=resnet.resnet34(pretrained=True)
        elif num_layers==50:
            base=resnet.resnet50(pretrained=True)
        elif num_layers==101:
            base = resnet.resnet101(pretrained=True)

        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )

        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        #self.last_context_layer=conv_bn(64,64,1)


        for head in sorted(self.heads):
          num_output = self.heads[head]

          # fc = nn.Conv2d(
          #     in_channels=64,
          #     out_channels=num_output,
          #     kernel_size=1,
          #     stride=1,
          #     padding=0
          # )


          if head_conv > 0:
            fc = nn.Sequential(
                nn.Conv2d(64, head_conv,
                  kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, num_output,
                  kernel_size=1, stride=1, padding=0))
          else:
            fc = nn.Conv2d(
              in_channels=64,
              out_channels=num_output,
              kernel_size=1,
              stride=1,
              padding=0
          )
          self.__setattr__(head, fc)

          #init
        for head in self.heads:
              final_layer = self.__getattr__(head)
              for i, m in enumerate(final_layer.modules()):
                  if isinstance(m, nn.Conv2d):
                      if m.weight.shape[0] == self.heads[head]:
                          if 'hm' in head:
                              nn.init.constant_(m.bias, -2.19)
                          else:
                              nn.init.normal_(m.weight, std=0.001)
                              nn.init.constant_(m.bias, 0)


    def forward(self, x):
        # Initial block
        x = self.in_block(x)

        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder blocks
        d4 = e3 + self.decoder4(e4)
        d3 = e2 + self.decoder3(d4)
        d2 = e1 + self.decoder2(d3)
        d1 = x + self.decoder1(d2)

        if self.is_train:
            ret = {}
            for head in self.heads:
                ret[head] = self.__getattr__(head)(d1)
            return [ret]
        else:
            ret = []
            for head in self.heads:
                ret.append(self.__getattr__(head)(d1))
            return ret


def get_link_net(num_layers, heads, head_conv,is_train=True):
  model = ReLinkNet(num_layers, heads, head_conv,is_train)
  return model


