import torch
from torch import nn
import torch.nn.functional as F
import random


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
        self.layer3 = self._make_layer(BasicBlockDec, 64, num_Blocks[2], stride=2, scale_factor=(2, 3))
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=1)
        self.layer1 = self._make_layer(BasicBlockDec, 32, num_Blocks[0], stride=2, scale_factor=(2, 2))
        self.conv1 = ResizeConv2d(32, nc, kernel_size=3, scale_factor=(2, 2))

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
        x = F.interpolate(x, scale_factor=(8,6))
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = x.view(x.size(0), 1, 256, 432)
        x = x[:, :, :, :431]
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


class VAE(nn.Module):
    def __init__(self, z_dim=10, output_units=20):
        super().__init__()
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


class Audio_Discriminator_MusiCNN(nn.Module):
        def __init__(self, y_input_dim=256, timbral_k_height=[0.4, 0.7], temporal_k_width=[32, 64, 128],
                     filter_factor=1.6):
            super().__init__()
            self.output_shape= (96, 215)
            self.y_input_dim = y_input_dim
            self.filter_factor = filter_factor
            if not isinstance(timbral_k_height, list):
                timbral_k_height = [timbral_k_height]
            if not isinstance(temporal_k_width, list):
                temporal_k_width = [temporal_k_width]
            self.k_h = [int(self.y_input_dim * k) for k in timbral_k_height]
            self.k_w = temporal_k_width

            self.conv_layers = nn.ModuleList()
            self.batch_norm_layers = nn.ModuleList()
            self.pool_layers = nn.ModuleList()

            self.dis_conv1 = nn.Conv1d(in_channels=138, out_channels=32, kernel_size=7,padding=3)
            self.dis_batch_norm1 = nn.InstanceNorm1d(32)
            # LAYER 2
            nn.init.xavier_uniform_(self.dis_conv1.weight)
            self.dis_max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
            self.dis_conv2 = nn.Conv1d(32,
                                   32,
                                   kernel_size=7,
                                   padding=3)
            self.dis_max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
            self.dis_batch_norm2 = nn.InstanceNorm1d(32)

            # LAYER 3
            nn.init.xavier_uniform_(self.dis_conv2.weight)
            self.dis_conv3 = nn.Conv1d(32,
                                   32,
                                   kernel_size=7,
                                   padding=3)
            self.dis_max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
            self.dis_batch_norm3 = nn.InstanceNorm1d(32)

            for k_h in self.k_h:
                conv_layer, batch_norm_layer, pool_layer = self.timbral_block(k_h)
                self.conv_layers.append(conv_layer)
                self.batch_norm_layers.append(batch_norm_layer)
                self.pool_layers.append(pool_layer)

            for k_w in self.k_w:
                conv_layer, batch_norm_layer, pool_layer = self.temporal_block(k_w)
                self.conv_layers.append(conv_layer)
                self.batch_norm_layers.append(batch_norm_layer)
                self.pool_layers.append(pool_layer)

        def timbral_block(self, k_h):
            out_channels = int(self.filter_factor * 32)
            conv_layer = nn.Conv2d(in_channels=1,
                                   out_channels=out_channels,
                                   kernel_size=(k_h, 8),
                                   padding=(0, 3))
            batch_norm = nn.BatchNorm2d(num_features=out_channels)
            h_out = self.y_input_dim - k_h + 1
            pool_layer = nn.MaxPool2d(kernel_size=(h_out, 2),
                                      stride=(h_out, 2))

            return conv_layer, batch_norm, pool_layer

        def temporal_block(self, k_w):
            out_channels = int(self.filter_factor * 8)
            pad_w = (k_w - 1) // 2
            conv_layer = nn.Conv2d(in_channels=1,
                                   out_channels=out_channels,
                                   kernel_size=(1, k_w),
                                   padding=(0, pad_w))
            batch_norm = nn.BatchNorm2d(num_features=out_channels)
            pool_layer = nn.MaxPool2d(kernel_size=(self.y_input_dim, 2),
                                      stride=(self.y_input_dim, 2))

            return conv_layer, batch_norm, pool_layer

        def forward(self, x):
            out = []
            for i in range(len(self.conv_layers)):
                conv = F.relu(self.conv_layers[i](x))
                bn = self.batch_norm_layers[i](conv)
                pool = self.pool_layers[i](bn)
                out.append(pool)
            out = torch.cat(out, dim=1)
            out = torch.squeeze(out)

            out_conv1 = F.relu(self.dis_conv1(out))
            out_bn_conv1 = self.dis_batch_norm1(out_conv1)

            out_conv2 = F.relu(self.dis_conv2(out_bn_conv1))
            out_bn_conv2 = self.dis_batch_norm2(out_conv2)
            res_conv2 = out_conv2 + out_bn_conv1

            out_conv3 = F.relu(self.dis_conv3(out_bn_conv2))
            out_bn_conv3 = self.dis_batch_norm3(out_conv3)
            res_conv3 = res_conv2 + out_bn_conv3
            out = torch.cat((out_bn_conv1, res_conv2, res_conv3), dim=1)

            return out


class Audio_Discriminator(nn.Module):
    def __init__(self, input_shape=[1, 256, 431]):
        super(Audio_Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)