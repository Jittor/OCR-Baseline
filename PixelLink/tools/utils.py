import os
from config_pl import cfg
import numpy as np
from PIL import Image


def output_prediction(prediction, filename):
    dir_path = os.path.join(cfg.save_dir, "prediction")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.join(dir_path, filename)
    file = open(file_path, "w", encoding='utf-8')
    for array in prediction:
        array_str = [str(i) for i in array.reshape([-1])]
        line = ",".join(array_str) + "\n"
        file.write(line)
    file.close()


def resize_prediction(prediction, img_path):
    image = np.array(Image.open(img_path))
    ori_h, ori_w = image.shape[:2]
    ratio_w = ori_w * 1.0 / cfg.input_size[0]
    ratio_h = ori_h * 1.0 / cfg.input_size[1]
    bboxes = np.array(prediction).reshape([-1, 4, 2]).astype(np.float32)
    bboxes[:, :, 1] *= ratio_h
    bboxes[:, :, 0] *= ratio_w
    recover_prediction = bboxes.reshape([-1, 8]).astype(int)
    return recover_prediction
