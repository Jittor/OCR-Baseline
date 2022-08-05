import jittor as jt
from jittor import nn
from jittor.models import vgg16_bn


def conv1x1_bn_relu(in_planes, out_planes, stride=1, has_bias=False):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                  padding=0, bias=has_bias),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(),
    )


def conv3x3_bn_relu(in_planes, out_planes, kernel_size=1, stride=1, padding=1, has_bias=False):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, bias=has_bias),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(),
    )


class VGG16(nn.Module):

    def __init__(self, pretrained):
        super(VGG16, self).__init__()
        self.backbone = vgg16_bn(pretrained=pretrained).features
        self.fc = nn.Sequential(
            conv3x3_bn_relu(512, 1024, 3, 1, 1),
            conv3x3_bn_relu(1024, 1024, 1, 1, 0))

    def execute(self, imgs):
        c1 = self.backbone[0](imgs)
        for i in range(1, 7):
            c1 = self.backbone[i](c1)
        c2 = c1
        for i in range(7, 13):
            c2 = self.backbone[i](c2)
        c3 = c2
        for i in range(13, 23):
            c3 = self.backbone[i](c3)
        c4 = c3
        for i in range(23, 33):
            c4 = self.backbone[i](c4)
        c5 = c4
        for i in range(33, 43):
            c5 = self.backbone[i](c5)
        c6 = self.fc(c5)
        return c1, c2, c3, c4, c5, c6


class VGGPixel(nn.Module):
    def __init__(self, pretrained=False, vggtype='2s'):
        super(VGGPixel, self).__init__()
        self.backbone = VGG16(pretrained)
        self.vggtype = vggtype
        self.cls_conv_6 = conv1x1_bn_relu(1024, 2)
        self.cls_conv_5 = conv1x1_bn_relu(512, 2)
        self.cls_conv_4 = conv1x1_bn_relu(512, 2)
        self.cls_conv_3 = conv1x1_bn_relu(256, 2)
        self.cls_conv_2 = conv1x1_bn_relu(128, 2)

        self.link_conv_6 = conv1x1_bn_relu(1024, 16)
        self.link_conv_5 = conv1x1_bn_relu(512, 16)
        self.link_conv_4 = conv1x1_bn_relu(512, 16)
        self.link_conv_3 = conv1x1_bn_relu(256, 16)
        self.link_conv_2 = conv1x1_bn_relu(128, 16)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear')
        if not pretrained:
            self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                jt.init.xavier_uniform_(m.weight)

    def execute(self, imgs):
        _, c2, c3, c4, c5, c6 = self.backbone(imgs)
        score_5 = self.cls_conv_6(c6)+self.cls_conv_5(c5)
        score_4 = self.cls_conv_4(c4)+self.upsample(score_5)
        if self.vggtype == '2s':
            score_3 = self.cls_conv_3(c3)+self.upsample(score_4)
            score_2 = self.cls_conv_2(c2)+self.upsample(score_3)
        else:
            score_2 = self.cls_conv_2(c3)+self.upsample(score_4)

        link_5 = self.link_conv_6(c6)+self.link_conv_5(c5)
        link_4 = self.link_conv_4(c4)+self.upsample(link_5)
        if self.vggtype == '2s':
            link_3 = self.link_conv_3(c3)+self.upsample(link_4)
            link_2 = self.link_conv_2(c2)+self.upsample(link_3)
        else:
            link_2 = self.link_conv_2(c3)+self.upsample(link_4)

        return score_2, link_2


if __name__ == '__main__':
    model = VGGPixel(pretrained=True)
    x = jt.random([1, 3, 224, 224])
    score, link = model(x)
    print(score, score.shape, link, link.shape)
