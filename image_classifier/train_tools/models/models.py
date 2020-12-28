import torch
import torch.nn as nn

__all__ = ['resnet8', 'resnet14', 'resnet20', 'TestCNN']

    
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Block(nn.Module):
    __constants__ = ['downsample']
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, 
                 base_width=64, dilation=1, norm_layer=None, expansion=1, block_type='basic'):
        super(Block, self).__init__()

        self.block_type = block_type
        self.expansion = expansion
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if block_type == 'basic':
            if groups != 1 or base_width != 64:
                raise ValueError('BasicBlock only supports groups=1 and base_width=64')
            if dilation > 1:
                raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        width = int(planes * (base_width / 64.)) * groups
        # Both conv3*3 with stride and self.downsample layers downsample the input when stride != 1
        if block_type == 'basic':
            self.conv1 = conv3x3(inplanes, width, stride)
            self.conv2 = conv3x3(width, width)

        self.bn1 = norm_layer(width)
        self.bn2 = norm_layer(width)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        if self.block_type == 'bottleneck':
            out = self.relu(out)
            
            out = self.conv3(out)
            out = self.bn3(out)
            
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out
    
    
class ResNet(nn.Module):
    BasicBlock_arch = ['resnet8', 'resnet14', 'resnet18', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']


    def __init__(self, arch, repeats, dim_in=3, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 32
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        if arch in self.BasicBlock_arch:
            self.expansion = 1
            self.block_type = 'basic'

        else:
            raise NotImplementedError('%s arch is not supported in ResNet' % arch)
        
        self.conv1 = nn.Conv2d(dim_in, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        planes = [64, 64, 128]
        self.planes = [p * self.expansion for p in planes]
        strides = [1, 2, 2]
        self.block_layers = self._make_layer(planes, repeats, strides)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(planes[-1] * self.expansion, num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Block):
                    if self.block_type == 'basic':
                        nn.init.constant_(m.bn2.weight, 0)

        print("ResNet was made")

    def _make_layer(self, planes, repeats, strides):        
        block_layers = []
        norm_layer = self._norm_layer
        for i in  range(3):
            plane = planes[i]
            repeat = repeats[i]
            stride = strides[i]
            
            downsample = None
            if stride != 1 or self.inplanes != plane * self.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, plane * self.expansion, stride),
                    norm_layer(plane * self.expansion),
                )

            layers = []
            layers.append(Block(self.inplanes, plane, stride, downsample, self.groups,
                                self.base_width, self.dilation, norm_layer, self.expansion, 
                                self.block_type))
            self.inplanes = plane * self.expansion
            for _ in range(1, repeat):
                layers.append(Block(self.inplanes, plane, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer, expansion=self.expansion, 
                                    block_type=self.block_type))
            block_layers.append(nn.Sequential(*layers))
        
        return nn.Sequential(*block_layers)
    
    def conv_stem(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        return x
    
    def pool_linear(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features = x
        x = self.classifier(x)
        
        return x, features
        
    def forward(self, x, get_features=False):
        x = self.conv_stem(x)
        
        x = self.block_layers(x)

        x, features = self.pool_linear(x)

        if get_features:
            return x, features
        
        else:
            return x
    
def _resnet(arch, repeats, **kwargs):
    model = ResNet(arch, repeats, **kwargs)
    return model


def resnet8(**kwargs):
    return _resnet('resnet8', [1, 1, 1], **kwargs)

def resnet14(**kwargs):
    return _resnet('resnet14', [2, 2, 2], **kwargs)

def resnet20(**kwargs):
    return _resnet('resnet20', [3, 3, 3], **kwargs)



class TestCNN(nn.Module):
    def __init__(self, dim_in=1, num_classes=20):
        super(TestCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=dim_in, out_channels=16, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, get_features=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x)))) # [N, 16, 12, 12]
        x = self.maxpool(self.relu(self.bn2(self.conv2(x)))) # [N, 32, 6, 6]
        x = self.avgpool(self.relu(self.conv3(x))) # [N, 64, 1, 1]
        
        features = x.view(x.size(0), -1)
        
        x = self.fc(features)
        
        if get_features:
            return x, features
        
        else:
            return x