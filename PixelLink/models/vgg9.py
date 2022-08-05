import jittor as jt
from jittor import nn


class VGG9bn(nn.Module):

    def __init__(self):
        super(VGG9bn, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm(64),
            nn.ReLU()
        )

        self.maxpool = nn.Pool(kernel_size=3, stride=2,
                               padding=1, op="maximum")
        self.layer1 = self.make_layers(64, [64, 64])
        self.maxpool_1 = nn.Pool(
            kernel_size=3, stride=2, padding=1, op="maximum")

        self.layer2 = self.make_layers(64, [128, 128])
        self.maxpool_2 = nn.Pool(
            kernel_size=3, stride=2, padding=1, op="maximum")

        self.layer3 = self.make_layers(128, [256, 256])
        self.maxpool_3 = nn.Pool(
            kernel_size=3, stride=2, padding=1, op="maximum")

        self.layer4 = self.make_layers(256, [512, 512])

    def make_layers(self, inchannels, channels_list, batch_norm=True, stride=1):
        layers = []
        conv2d = nn.Conv2d(
            inchannels, channels_list[0], kernel_size=3, padding=1, stride=stride)
        layers += [conv2d,
                   nn.BatchNorm(channels_list[0]), nn.ReLU()]
        inchannels = channels_list[0]
        for v in channels_list[1:]:
            conv2d = nn.Conv2d(inchannels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            inchannels = v
        return nn.Sequential(*layers)

    def execute(self, x):
        blocks = []
        x = self.layer0(x)  # 2
        x = self.maxpool(x)  # 4

        x = self.layer1(x)
        blocks.append(x)
        x = self.maxpool_1(x)  # 4

        x = self.layer2(x)
        blocks.append(x)
        x = self.maxpool_2(x)  # 8

        x = self.layer3(x)
        blocks.append(x)
        x = self.maxpool_3(x)  # 16

        x = self.layer4(x)
        blocks.append(x)

        return x, blocks


class VGG9FPN(nn.Module):
    def __init__(self, feat_stride=4):
        super(VGG9FPN, self).__init__()
        self.feat_stride = feat_stride
        self.resnet = VGG9bn()
        self.conv1 = nn.Conv2d(512+256, 128, 1)
        self.bn1 = nn.BatchNorm(128)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm(128)
        self.relu2 = nn.ReLU()
        #self.unpool1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        #self.unpool2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = nn.Conv2d(128+128, 64, 1)
        self.bn3 = nn.BatchNorm(64)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm(64)
        self.relu4 = nn.ReLU()
        #self.unpool3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv5 = nn.Conv2d(64+64, 64, 1)
        self.bn5 = nn.BatchNorm(64)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn6 = nn.BatchNorm(32)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn7 = nn.BatchNorm(32)
        self.relu7 = nn.ReLU()

        self.cls_head = self._make_head(32, 2)
        self.link_head = self._make_head(32, 16)
        self.count = 0
        self.k = 10
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def _make_head(self, in_planes, out_planes):
        layers = []
        for _ in range(2):
            layers.append(
                nn.Conv2d(in_planes, 32, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            in_planes = 32
        layers.append(nn.Conv2d(32, out_planes,
                      kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def execute(self, input):
        _, f = self.resnet(input)
        x_32 = f[3]  # bs 2048 w/32 h/32

        g = (self.upsample(x_32))  # bs 2048 w/16 h/16
        c = self.conv1(jt.concat((g, f[2]), 1))
        c = self.bn1(c)
        c = self.relu1(c)

        h = self.conv2(c)  # bs 128 w/16 h/16
        h = self.bn2(h)
        x_16 = self.relu2(h)

        g = self.upsample(x_16)
        c = self.conv3(jt.concat((g, f[1]), 1))
        c = self.bn3(c)
        c = self.relu3(c)

        h = self.conv4(c)  # bs 64 w/8 h/8
        h = self.bn4(h)
        x_8 = self.relu4(h)

        g = self.upsample(x_8)
        c = self.conv5(jt.concat((g, f[0]), 1))
        c = self.bn5(c)
        c = self.relu5(c)

        h = self.conv6(c)  # bs 32 w/4 h/4
        h = self.bn6(h)
        h = self.relu6(h)
        g = self.conv7(h)  # bs 32 w/4 h/4
        g = self.bn7(g)
        x_4 = self.relu7(g)

        cls_preds = self.cls_head(x_4)
        link_preds = self.link_head(x_4)

        out_put = [cls_preds, link_preds]

        return out_put
