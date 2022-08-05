import os
import re
import numpy as np
from PIL import Image
from jittor import Var
import jittor as jt
from numpy import ndarray
import config


def not_real(num):
    if isinstance(num, Var) or isinstance(num, ndarray):
        num = num.item()
    return num != num or num == float("inf") or num == float("-inf")


def get_alphabet(path=config.alphabet_path):
    alphabet = str()
    if os.path.exists(path):
        # load alphabet
        with open(path, encoding='utf-8') as f:
            data = f.readlines()
            alphabet = [x.rstrip() for x in data]
            alphabet = ''.join(alphabet)
    return alphabet


def save_model(model, save_dir=config.save_dir, save_file=f'crnn_final.pkl'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_model_path = os.path.join(save_dir, save_file)
    model.save(save_model_path)
    print('Save model at ', save_model_path)


def get_xywh(gt):
    x = min(gt[0], gt[6])
    y = min(gt[1], gt[3])
    w = max(gt[2] - gt[0], gt[4] - gt[6], gt[2] - gt[6], gt[4] - gt[0])
    h = max(gt[7] - gt[1], gt[5] - gt[3], gt[7] - gt[3], gt[5] - gt[1])
    return x, y, w, h


def get_text_and_box(path, verify=False):
    texts = list()
    boxes = list()
    with open(path, encoding='utf-8') as file:
        lines = file.readlines()
    for line in lines:
        text = line.rstrip().split(",")[-1]
        box = line.rstrip().split(",")[:8]
        if verify and not valid_gt_text(text):
            continue
        texts.append(text)
        boxes.append(box)
    return texts, boxes


def process_img(image, img_width=config.img_width, img_height=config.img_height, img_channel=config.img_channel):
    image = image.resize((img_width, img_height), resample=Image.BILINEAR)
    image = np.array(image)
    image = image.reshape((img_channel, img_height, img_width))
    image = (image / 127.5) - 1.0
    image = jt.float32(image)
    return image


def valid_gt_text(text):
    return not (re.search(r"[\WA-Za-z]", text)) and len(text) >= 2


def a_match_b(a, b):
    match = 0
    for i in range(len(b)):
        index = a.index(b[i]) if b[i] in a else -1
        if index != -1:
            match += 1
            a = a[index + 1:]
    return match


def count_similarity(a, b):
    if a == b:
        return 1
    match1 = a_match_b(a, b)
    match2 = a_match_b(b, a)
    match = max(match1, match2)
    return match / (len(a) + len(b) - match)


if __name__ == '__main__':
    assert count_similarity("11", "1111") == 0.5
    assert count_similarity("121", "124") == 0.5
    assert count_similarity("121", "121111") == 0.5
    assert count_similarity("121", "112") == 0.5
    assert count_similarity("1234", "1235") == 3 / 5
    assert count_similarity("1234", "1267") == 2 / 6
    assert count_similarity("121", "1214") == 3 / 4
    assert count_similarity("125", "1214") == 2 / 5
    assert count_similarity("121", "11214") == 3 / 5
    assert count_similarity("121", "13214") == 3 / 5
    assert count_similarity([1, 2, 1], [1, 1, 2]) == 0.5
    assert count_similarity([1, 2, 1], [1, 1, 2, 1, 4]) == 3 / 5
    assert count_similarity(["1", "2", "1"], ["1", "3", "2", "1", "4"]) == 3 / 5

    assert valid_gt_text("test") is False
    assert valid_gt_text("你好吗？") is False
    assert valid_gt_text("你好.") is False
    assert valid_gt_text("你好") is True
    assert valid_gt_text("你") is False
    assert valid_gt_text("h你好") is False
