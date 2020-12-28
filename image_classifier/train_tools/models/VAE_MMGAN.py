import torch
from torch import nn, optim
import torch.nn.functional as F
import random

__all__ = ['ImgVAE', 'ImgDiscriminator', 'ReplayBuffer']

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1, scale_factor=1):
        super().__init__()

        planes = in_planes * stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride*scale_factor, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1 and scale_factor == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride*scale_factor, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1, scale_factor=1):
        super().__init__()

        planes = int(in_planes / stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1 and scale_factor == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=scale_factor)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=scale_factor),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet20Enc(nn.Module):

    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=10, nc=1):
        super().__init__()
        self.in_planes = 32
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=2)
        self.layer2 = self._make_layer(BasicBlockEnc, 64, num_Blocks[1], stride=1)
        self.layer3 = self._make_layer(BasicBlockEnc, 128, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 128, num_Blocks[3], stride=1, scale_factor=2)
        self.vae_fc = nn.Linear(128, 2 * z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride, scale_factor=1):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride, scale_factor)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        feature = x.view(x.size(0), -1)
        x = self.vae_fc(feature)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        return mu, logvar, feature

    
class ResNet20Dec(nn.Module):

    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=10, nc=1):
        super().__init__()
        self.in_planes = 128

        self.linear = nn.Linear(z_dim, 128)

        self.layer4 = self._make_layer(BasicBlockDec, 128, num_Blocks[3], stride=1)
        self.layer3 = self._make_layer(BasicBlockDec, 64, num_Blocks[2], stride=2, scale_factor=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=1)
        self.layer1 = self._make_layer(BasicBlockDec, 32, num_Blocks[0], stride=2, scale_factor=2)
        self.conv1 = ResizeConv2d(32, nc, kernel_size=7, scale_factor=1)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride, scale_factor=1):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride, scale_factor)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 128, 1, 1)
        x = F.interpolate(x, scale_factor=2)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = x.view(x.size(0), 1, 28, 28)
        return x


class ReplayBuffer:
    def __init__(self, max_size=50):

        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


class ImgVAE(nn.Module):

    def __init__(self, z_dim=10, output_units=20):
        super(ImgVAE, self).__init__()
        self.encoder = ResNet20Enc(z_dim=z_dim)
        self.decoder = ResNet20Dec(z_dim=z_dim+output_units+128)
        self.classifier = nn.Linear(z_dim, output_units)

    def forward(self, x, is_training=True, prior=None, original_feature=False):
        mean, logvar, feature = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        cls = self.classifier(feature)
        if is_training:
            assert prior is not None
            x = self.decoder(torch.cat((feature, prior, z), dim=1))
        else:
            x = self.decoder(torch.cat((feature, cls, z), dim=1))
        if original_feature:
            return x, z, mean, logvar, cls, feature
        else:
            return x, z, mean, logvar, cls

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2)  # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

    
# Discriminator model
class ImgDiscriminator(torch.nn.Module):
    def __init__(self, input_dim=1, num_filters=[64, 64, 64, 64], output_dim=1):
        super(ImgDiscriminator, self).__init__()
        self.hidden_layer1 = torch.nn.Sequential()
        self.hidden_layer2 = torch.nn.Sequential()
        self.hidden_layer = torch.nn.Sequential()
        for i in range(len(num_filters)):
            # Convolutional layer
            if i == 0:
                # For input
                input_conv = torch.nn.Conv2d(input_dim, int(num_filters[i]), kernel_size=3, stride=1, padding=1)
                self.hidden_layer1.add_module('input_conv', input_conv)

                # Initializer
                torch.nn.init.normal_(input_conv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(input_conv.bias, 0.0)

                # Activation
                self.hidden_layer1.add_module('label_act', torch.nn.LeakyReLU(0.2))
            else:
                conv = torch.nn.Conv2d(num_filters[i-1], num_filters[i], kernel_size=3, stride=2, padding=1)

                conv_name = 'conv' + str(i + 1)
                self.hidden_layer2.add_module(conv_name, conv)

                # Initializer
                torch.nn.init.normal_(conv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(conv.bias, 0.0)

                # Batch normalization
                bn_name = 'bn' + str(i + 1)
                self.hidden_layer2.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

                # Activation
                act_name = 'act' + str(i + 1)
                self.hidden_layer2.add_module(act_name, torch.nn.LeakyReLU(0.2))

        # Output layer
        self.output_layer = torch.nn.Sequential()
        # Convolutional layer
        out = torch.nn.Conv2d(num_filters[i], output_dim, kernel_size=4, stride=1, padding=0)
        self.output_layer.add_module('out', out)
        # Initializer
        torch.nn.init.normal_(out.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(out.bias, 0.0)
        # Activation
        self.output_layer.add_module('act', torch.nn.Sigmoid())

    def forward(self, z):
        h = self.hidden_layer1(z)
        h = self.hidden_layer2(h)
        out = self.output_layer(h).view(z.size(0), -1)
        return out
    
