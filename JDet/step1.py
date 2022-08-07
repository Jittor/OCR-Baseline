from jdet.runner import Runner
from jdet.config import init_cfg, update_cfg, get_cfg
from PIL import Image
import jittor as jt
from jdet.data.transforms import Compose
import numpy as np

jt.flags.use_cuda = 1


def sort_point(gt):
    def take_first(elem):
        return elem[0]
    gt_ = [[gt[i], gt[i+1]] for i in [0, 2,  4, 6]]
    gt_.sort(key=take_first)
    if gt_[0][1] > gt_[1][1]:
        p1 = gt_[1]
        p4 = gt_[0]
    else:
        p1 = gt_[0]
        p4 = gt_[1]
    if gt_[2][1] > gt_[3][1]:
        p2 = gt_[3]
        p3 = gt_[2]
    else:
        p2 = gt_[2]
        p3 = gt_[3]
    return [p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1]]


def get_xywh(gt):
    x = min(gt[0], gt[6])
    y = min(gt[1], gt[3])
    w = max(gt[2]-gt[0], gt[4]-gt[6], gt[2]-gt[6], gt[4]-gt[0])
    h = max(gt[7]-gt[1], gt[5]-gt[3], gt[7]-gt[3], gt[5]-gt[1])
    return max(int(x), 0), max(int(y), 0), int(w), int(h)


class Step1():
    def __init__(self, config_path, weights_path) -> None:
        init_cfg(config_path)
        cfg = get_cfg()
        update_cfg({"resume_path": weights_path})
        self.runner = Runner()
        self.transforms = Compose(cfg.dataset.val.transforms)

    def infer(self, img):
        iw, ih = img.size
        targets = dict(
            ori_img_size=img.size,
            img_size=img.size,
            scale_factor=1.,
            img_file='None'
        )

        if self.transforms:
            image, target = self.transforms(img, targets)
        image = jt.Var(image).unsqueeze(0)
        results = self.runner.run_on_one_image(image, [target])[0][0].numpy()
        results[:, 0::2] = np.clip(results[:, 0::2], 0, iw-1)
        results[:, 1::2] = np.clip(results[:, 1::2], 0, ih-1)
        return results


if __name__ == "__main__":
    step1 = Step1('store_sign_detection/s2anet_r50_fpn_5x_ocr_630_1120_bs4.py',
                  'work_dirs/s2anet_r50_fpn_5x_ocr_630_1120_bs4/checkpoints/ckpt_43.pkl')
    img = Image.open(
        '/home/data/1305/011284_1508290961970980.jpg').convert("RGB")
    results = step1.infer(img)
    iw, ih = img.size
    for i, result in enumerate(results):
        x, y, w, h = get_xywh(sort_point(result.tolist()))
        if x > iw or y > ih:
            continue
        img_crop = Image.fromarray(
            np.array(img)[y:min(y + h, ih), x:min(x + w, iw)])
        img_crop.save('test{}.png'.format(i))
