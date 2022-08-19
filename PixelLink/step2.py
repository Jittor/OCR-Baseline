import numpy as np
from PIL import Image
import jittor as jt
import os
import sys

curr_path = os.path.dirname(__file__)
sys.path.extend([
    os.path.join(curr_path, '../'),
    ])
from PixelLink.models.vgg import VGGPixel
from PixelLink.tools import postprocess
from PixelLink.dataset.transform import build_transform
from PixelLink.config_pl import cfg


def get_xywh(gt):
    x = min(gt[0], gt[6])
    y = min(gt[1], gt[3])
    w = max(gt[2] - gt[0], gt[4] - gt[6], gt[2] - gt[6], gt[4] - gt[0])
    h = max(gt[7] - gt[1], gt[5] - gt[3], gt[7] - gt[3], gt[5] - gt[1])
    return max(int(x), 0), max(int(y), 0), int(w), int(h)


def resize_predict(prediction, image):
    if not prediction:
        return prediction
    ori_w, ori_h = image.size
    ratio_w = ori_w * 1.0 / cfg.input_size[0]
    ratio_h = ori_h * 1.0 / cfg.input_size[1]
    prediction = np.array(prediction).astype(np.float32)
    prediction[:, :, 1] *= ratio_h
    prediction[:, :, 0] *= ratio_w
    return prediction


class Step2:
    def __init__(self, ckp_path) -> None:
        self.model = VGGPixel()
        self.model.eval()
        self.model.load(ckp_path)
        self.val_transform = build_transform(cfg, mode="val")

    def infer(self, image):
        t_image, _ = self.val_transform(np.array(image), None)
        t_image = jt.Var(t_image).unsqueeze(0)
        out_1, out_2 = self.model(t_image)

        predict = postprocess.cal_label_on_batch(out_1, out_2)[0]
        r_predict = resize_predict(predict, image)
        return r_predict


if __name__ == "__main__":
    step2 = Step2('/project/train/models/PixelLink.pkl')
    img = Image.open('/home/data/1414/016201_1508424072625001_1.jpg').convert("RGB")
    results = step2.infer(img)
    for i, result in enumerate(results):
        iw, ih = img.size
        x, y, w, h = get_xywh(result.reshape(-1).tolist())
        if x > iw or y > ih or w == 0 or h == 0:
            continue
        img_crop = Image.fromarray(
            np.array(img)[y:min(y + h, ih), x:min(x + w, iw)])
        img_crop.save('test{}.jpg'.format(i))
