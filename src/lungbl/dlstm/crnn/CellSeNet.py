import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch.nn.functional as F
import torch
import sys
import lungbl.dlstm.crnn as crnn
#from .resnet import ResNet
#from .senet.se_module import SELayer

# from https://github.com/moskomule/senet.pytorch/blob/master/senet/se_resnet.py

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 1 # used to be expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction) # used to be planes * 4
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
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        #print (residual.shape, out.shape)
        out += residual
        out = self.relu(out)

        return out

class CellResNet(nn.Module):  # haven't test. 20190505

    def __init__(self,
                 block,
                 layers,
                 max_step,
                 sample_size,
                 sample_duration,
                 in_channel,
                 dropout = False,
                 shortcut_type='B',
                 num_classes=4_00, style = 'Front'):
        self.inplanes = 64
        super(CellResNet, self).__init__()
        self.style = style
        self.max_step = max_step
        self.lstm3d = crnn.Conv3dLSTMCell(in_channels = in_channel, out_channels = in_channel * max_step, kernel_size = 3)
        self.lstm2d = crnn.Conv3dLSTMCell(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.lstm1d = crnn.Conv1dLSTMCell(in_channels = in_channel, out_channels = 128, kernel_size = 3)
        self.conv1 = nn.Conv3d(
            in_channel,                                                   # here used to be 3, my channel is 2, so here be 2
            64,
            kernel_size=7,
            stride=(2, 2, 2),                                   # here used to be [1,2,2], but my size is [128, 128, 128], dims are same
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 32, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 64, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 128, layers[3], shortcut_type, stride=2)
#         last_duration = int(math.ceil(sample_duration / 16))
#         last_size = int(math.ceil(sample_size / 32))
#         self.avgpool = nn.AvgPool3d(
#             last_size, stride=1)                 # use to be (last_duration, last_size, last_size)
        #print ('last duration, last size: ', last_duration, last_size)
        self.fc = nn.Linear(128, num_classes) #* block.expansion
        self.dropout = dropout
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.style == 'Front':
            for i in range(self.max_step):
                #print ('x[i].shape', x[i].shape)
                if i == 0:
                    hx, cx = self.lstm3d(x[i])
                    #print (hx.shape, cx.shape, x.shape)
                else:
                    hx, cx = self.lstm3d(x[i], (hx, cx))
            #print ('hx shape', hx.shape)
            x = self.conv1(hx)
            print (x.shape)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            #print (x.shape, '------------')
            x = F.adaptive_avg_pool3d(x, 1)
            #print (x.shape, '------------')
            x = x.view(x.size(0), -1)
            if self.dropout:
                x = nn.Dropout(0.5)(x)
            x = self.fc(x)
            
        if self.style == 'Middle':
            for i in range(self.max_step):
                #print (x[i].shape, x.shape)
                tmp_x = self.conv1(x[i])

                tmp_x = self.bn1(tmp_x)
                tmp_x = self.relu(tmp_x)
                tmp_x = self.maxpool(tmp_x)

                tmp_x = self.layer1(tmp_x)
                tmp_x = self.layer2(tmp_x)
                if i == 0:
                    hx, cx = self.lstm2d(tmp_x)
                    #print (hx.shape, cx.shape, x.shape)
                else:
                    hx, cx = self.lstm2d(tmp_x, (hx, cx))
            x = self.layer3(hx)
            x = self.layer4(x)
            #print (x.shape, '------------')
            x = F.adaptive_avg_pool3d(x, 1)
            #print (x.shape, '------------')
            x = x.view(x.size(0), -1)
            if self.dropout:
                x = nn.Dropout(0.5)(x)
            x = self.fc(x)
            
        if self.style == 'Bottom':
            for i in range(self.max_step):
                x[i] = self.conv1(x[i])
                x[i] = self.bn1(x[i])
                x[i] = self.relu(x[i])
                x[i] = self.maxpool(x[i])

                x[i] = self.layer1(x[i])
                x[i] = self.layer2(x[i])
                x[i] = self.layer3(x[i])
                x[i] = self.layer4(x[i])
                #print (x.shape, '------------')
                x[i] = F.adaptive_avg_pool3d(x[i], 1)
                #print (x.shape, '------------')
                x[i] = x[i].view(x[i].size(0), -1)
                if self.dropout:
                    x[i] = nn.Dropout(0.5)(x[i])
                x[i] = self.fc(x[i])
                if i == 0:
                    hx, cx = self.lstm1d(x[i])
                    #print (hx.shape, cx.shape, x.shape)
                else:
                    hx, cx = self.lstm1d(x[i], (hx, cx))
            x = hx
        return 0, x


    
def cellsenet18(**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = CellResNet(SEBasicBlock, [2, 2, 2, 2], **kwargs)
    model.avgpool = nn.AdaptiveAvgPool3d(1)
    return model


def se_resnet34(num_classes=1_000, sample_size = 1_28, sample_duration = 1_6, in_channel = 1_5, dropout = False):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes, sample_size = sample_size, sample_duration = sample_duration, in_channel = in_channel)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet50(num_classes=1_000, pretrained=False, sample_size = 1_28, sample_duration = 1_6, in_channel = 1_5, dropout = False):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes, sample_size = sample_size, sample_duration = sample_duration, in_channel = in_channel)
    model.avgpool = nn.AdaptiveAvgPool3d(1)
    if pretrained:
        model.load_state_dict(model_zoo.load_url("https://www.dropbox.com/s/xpq8ne7rwa4kg4c/seresnet50-60a8950a85b2b.pkl"))
    return model


def se_resnet101(num_classes=1_000, pretrained=False, sample_size = 1_28, sample_duration = 1_6, in_channel = 1_5, dropout = False):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes, sample_size = sample_size, sample_duration = sample_duration, in_channel = in_channel, dropout = dropout)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet152(num_classes=1_000):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


class CifarSEBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super(CifarSEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += residual
        out = self.relu(out)

        return out


class CifarSEResNet(nn.Module):
    def __init__(self, block, n_size, num_classes=10, reduction=16):
        super(CifarSEResNet, self).__init__()
        self.inplane = 16
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, blocks=n_size, stride=1, reduction=reduction)
        self.layer2 = self._make_layer(block, 32, blocks=n_size, stride=2, reduction=reduction)
        self.layer3 = self._make_layer(block, 64, blocks=n_size, stride=2, reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride, reduction))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class CifarSEPreActResNet(CifarSEResNet):
    def __init__(self, block, n_size, num_classes=10, reduction=16):
        super(CifarSEPreActResNet, self).__init__(block, n_size, num_classes, reduction)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.initialize()

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)


def se_resnet20(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = CifarSEResNet(CifarSEBasicBlock, 3, **kwargs)
    return model


def se_resnet32(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarSEResNet(CifarSEBasicBlock, 5, **kwargs)
    return model


def se_resnet56(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarSEResNet(CifarSEBasicBlock, 9, **kwargs)
    return model


def se_preactresnet20(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = CifarSEPreActResNet(CifarSEBasicBlock, 3, **kwargs)
    return model


def se_preactresnet32(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarSEPreActResNet(CifarSEBasicBlock, 5, **kwargs)
    return model


def se_preactresnet56(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarSEPreActResNet(CifarSEBasicBlock, 9, **kwargs)
    return model