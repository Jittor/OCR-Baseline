import numpy as np
from numpy import random
from .utils import imresize, bgr2hsv, hsv2bgr, crop
from PIL import Image
import cv2
import jittor.transform as transform


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, gt=None):
        for t in self.transforms:
            image, gt = t(image, gt)

        return image, gt


class PhotoMetricDistortion(object):
    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(2):
            img = bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(2):
            img = bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = hsv2bgr(img)
        return img

    def __call__(self, image, gt=None):
        # random brightness
        image = self.brightness(image)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            image = self.contrast(image)

        # random saturation
        image = self.saturation(image)

        # random hue
        image = self.hue(image)

        # random contrast
        if mode == 0:
            image = self.contrast(image)

        return image, gt


class Resize(object):
    def __init__(self, size=(512, 512)):
        self.size = size

    def __call__(self, image, gt=None):
        resized_image = imresize(image, self.size)
        if gt:
            bboxes = np.array(gt["bboxes"]).reshape(
                [-1, 4, 2]).astype(np.float32)
            ori_h, ori_w = image.shape[:2]
            ratio_w = self.size[0]*1.0/ori_w
            ratio_h = self.size[1]*1.0/ori_h
            bboxes[:, :, 1] *= ratio_h
            bboxes[:, :, 0] *= ratio_w
            gt["bboxes"] = bboxes.reshape([-1, 8]).astype(int).tolist()
        return resized_image, gt


class FilterBboxes(object):
    def __init__(self):
        pass

    def __call__(self, image, gt=None, method="rai"):
        def distance(a, b):
            return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

        def min_side_ignore(label):
            label = np.array(label).reshape(4, 2)
            dists = []
            for i in range(4):
                dists.append(distance(label[i], label[(i+1) % 4]))
            if min(dists) < 10:
                return True  # ignore it
            else:
                return False

        def remain_area_ignore(label, origin_area):
            label = np.array(label).reshape(4, 2)
            area = cv2.contourArea(label)
            if area / origin_area < 0.2:
                return True
            else:
                return False
        if method == "msi":
            ignore = list(map(min_side_ignore, gt["bboxes"]))
        elif method == "rai":
            ignore = list(
                map(remain_area_ignore, gt["bboxes"], gt["area"]))
        else:
            ignore = [False] * 8
        gt["ignore"] = list(
            map(lambda a, b: a or b, gt["ignore"], ignore))
        return image, gt


class RandomCrop(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, gt=None):
        if random.random() > self.prob:
            return image, gt
        scale = 0.1 + random.random() * 0.9
        image, img_range = crop(image, scale=scale)

        new_h = img_range[3] - img_range[1]
        new_w = img_range[2] - img_range[0]
        new_bboxes = np.array(gt["bboxes"]).reshape(
            [-1, 4, 2]).astype(np.float32)
        new_bboxes[:, :, 1] -= img_range[1]
        new_bboxes[:, :, 0] -= img_range[0]
        new_bboxes[new_bboxes < 0] = 0
        new_bboxes[:, :, 1][new_bboxes[:, :, 1] >= new_h] = new_h - 1
        new_bboxes[:, :, 0][new_bboxes[:, :, 0] >= new_w] = new_w - 1
        gt["bboxes"] = new_bboxes.reshape([-1, 8]).astype(int).tolist()
        return image, gt


class RandomRotate(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, gt=None):
        if random.random() > self.prob:
            return image, gt
        image = Image.fromarray(np.uint8(image))
        origin_w, origin_h = image.size
        rand_angle = random.randint(0, 3) * 90
        image = np.array(image.rotate(rand_angle, expand=True))
        new_bboxes = np.array(gt["bboxes"]).reshape(
            [-1, 4, 2]).astype(np.float32)
        for i in range(rand_angle // 90):
            new_bboxes[:, :, 0] = origin_w - 1 - new_bboxes[:, :, 0]
            new_bboxes[:, :, [0, 1]] = new_bboxes[:, :, [1, 0]]
            origin_h, origin_w = origin_w, origin_h
        gt["bboxes"] = new_bboxes.reshape([-1, 8]).astype(int).tolist()
        return image, gt


class Normalize(object):
    def __init__(self, mean, std):
        self.norm = transform.ImageNormalize(mean, std)

    def __call__(self, image, gt=None):
        image = self.norm(image)
        return image, gt


class ToTensor(object):
    def __init__(self):
        self.to_tensor = transform.ToTensor()

    def __call__(self, image, gt=None):
        image = self.to_tensor(image).transpose(2, 0, 1)
        return image, gt


class build_transform(object):
    def __init__(self, cfg, mode="train"):
        if mode == 'train':
            self.aug = Compose([
                PhotoMetricDistortion(),
                RandomRotate(),
                # RandomCrop(),   # Todo
                FilterBboxes(),
                Resize(cfg.input_size),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
            ])
        else:
            self.aug = Compose([
                Resize(cfg.input_size),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
            ])

    def __call__(self, image, gt):
        return self.aug(image, gt)
