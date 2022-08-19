from argparse import ArgumentParser
import re
import os
from tqdm import tqdm
import numpy as np
import cv2
import json
from ji import init, process_image


def valid_gt_text(text):
    return not (re.search(r"[\WA-Za-z]", text)) and len(text) >= 2


def get_text_and_box(labels, verify=False):
    texts = list()
    boxes = list()
    for label in labels:
        text = label[-1]
        if verify and not valid_gt_text(text):
            continue
        texts.append(text)
        box = label[:8]
        box = [int(point) for point in box]
        boxes.append(box)
    return texts, boxes


def get_label_lines(path, verify=False):
    box_text = list()
    with open(path, encoding='utf-8') as file:
        lines = file.readlines()
    for line in lines:
        text = line.rstrip().split(",")[-1]
        if verify and not valid_gt_text(text):
            continue
        box = line.rstrip().split(",")[:8]
        box = [int(point) for point in box]
        box.append(text)
        box_text.append(box)
    return box_text


def match_text(text1, text2):
    match = 0
    for i in range(len(text2)):
        index = text1.index(text2[i]) if text2[i] in text1 else -1
        if index != -1:
            match += 1
            text1 = text1[index + 1:]
    return match


def count_similarity(text1, text2):
    if text1 == text2:
        return 1
    match1 = match_text(text1, text2)
    match2 = match_text(text2, text1)
    match = max(match1, match2)
    return match / (len(text1) + len(text2) - match)


def cal_iou(box1, box2):
    """
    box1, box2: list or numpy array of size 4*2 or 8, h_index first
    """
    box1 = np.array(box1).reshape([1, 4, 2])
    box2 = np.array(box2).reshape([1, 4, 2])
    box1_max = box1.max(axis=1)
    box2_max = box2.max(axis=1)
    w_max = max(box1_max[0][0], box2_max[0][0])
    h_max = max(box1_max[0][1], box2_max[0][1])
    canvas = np.zeros((h_max + 1, w_max + 1))
    box1_canvas = canvas.copy()
    box1_area = np.sum(cv2.drawContours(
        box1_canvas, box1, -1, 1, thickness=-1))
    box2_canvas = canvas.copy()
    box2_area = np.sum(cv2.drawContours(
        box2_canvas, box2, -1, 1, thickness=-1))
    cv2.drawContours(canvas, box1, -1, 1, thickness=-1)
    cv2.drawContours(canvas, box2, -1, 1, thickness=-1)
    union = np.sum(canvas)
    intersction = box1_area + box2_area - union
    return intersction / union


def evaluate(pred_json, gt_json, threshold_text, threshold_box):
    pred_dict = json.loads(pred_json)
    gt_dict = json.loads(gt_json)
    assert gt_dict.keys() == pred_dict.keys()
    img_ids = gt_dict.keys()

    correct_num = 0
    pred_num = 0
    gt_num = 0
    pbar = tqdm(enumerate(img_ids), total=len(img_ids), desc="Eval")
    for index, img_id in pbar:
        pbar.set_description("Evaluation - i:{}, {}".format(index + 1, img_id))

        gt_texts, gt_boxes = get_text_and_box(gt_dict[img_id], verify=True)
        pred_texts, pred_boxes = get_text_and_box(pred_dict[img_id])
        gt_num += len(gt_texts)
        pred_num += len(pred_texts)

        matched_j = list()
        for i in range(len(gt_texts)):
            for j in range(len(pred_texts)):
                if j in matched_j:
                    continue
                score_text = count_similarity(gt_texts[i], pred_texts[j])
                score_bbox = cal_iou(gt_boxes[i], pred_boxes[j])
                if score_text >= threshold_text and score_bbox >= threshold_box:  # correct prediction
                    correct_num += 1
                    matched_j.append(j)
    pbar.close()

    # overall result
    acc = round(correct_num / pred_num, 4) if pred_num else 0
    recall = round(correct_num / gt_num, 4) if gt_num else 0
    lines = list()
    lines.append("\n-------------------Evaluation-------------------")
    lines.append("gt_num: {}".format(gt_num))
    lines.append("pred_num: {}".format(pred_num))
    lines.append("correct_num: {}".format(correct_num))
    lines.append("acc: {}".format(acc))
    lines.append("recall: {}\n".format(recall))
    lines = "\n".join(lines)
    print(lines)
    return acc, recall


def get_gt_json(gt_dir):
    filenames = os.listdir(gt_dir)
    filenames = [filename for filename in filenames if ".txt" in filename]
    gt_paths = [os.path.join(gt_dir, filename) for filename in filenames]
    img_ids = [filename.split(".")[0] for filename in filenames]
    gt_dict = dict()
    pbar = tqdm(enumerate(img_ids), total=len(img_ids), desc="GET LABELS")
    for index, img_id in pbar:
        pbar.set_description("GET LABELS - i:{}, {}".format(index + 1, img_id))
        gt_path = gt_paths[index]
        gt_labels = get_label_lines(gt_path)
        gt_dict[img_id] = gt_labels
    gt_json = json.dumps(gt_dict)
    return gt_json


def get_pred_json(img_dir):
    model = init()
    filenames = os.listdir(img_dir)
    filenames = [filename for filename in filenames if ".jpg" in filename]
    img_paths = [os.path.join(img_dir, filename) for filename in filenames]
    img_ids = [filename.split(".")[0] for filename in filenames]
    pred_dict = dict()
    pbar = tqdm(enumerate(img_ids), total=len(img_ids), desc="Prediction")
    for index, img_id in pbar:
        pbar.set_description("Prediction - i:{}, {}".format(index + 1, img_id))
        img_path = img_paths[index]
        input_image = cv2.imread(img_path)
        img_pred_json = process_image(model, input_image, args=None)
        img_pred = json.loads(img_pred_json)
        pred_dict[img_id] = [i["points"] + [i["name"]]
                             for i in img_pred["model_data"]["objects"]]
    pred_json = json.dumps(pred_dict)
    return pred_json


def main():
    gt_json = get_gt_json(gt_dir=args.gt)
    pred_json = get_pred_json(img_dir=args.img)
    acc, recall = evaluate(
        pred_json, gt_json, threshold_text=args.threshold_text, threshold_box=args.threshold_box)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i",
                        "--img",
                        default="/home/data/1411",
                        type=str,
                        help="path of images.",
                        metavar="IMAGE PATH")
    parser.add_argument("-g",
                        "--gt",
                        default="/home/data/1411",
                        type=str,
                        help="path of ground truth.",
                        metavar="GT PATH")
    parser.add_argument("-t1",
                        "--threshold_text",
                        default=0.6,
                        type=float,
                        help="threshold for evaluate text.",
                        metavar="THRESHOLD TEXT")
    parser.add_argument("-t2",
                        "--threshold_box",
                        default=0.5,
                        type=float,
                        help="threshold for evaluate bounding box.",
                        metavar="THRESHOLD BOUNDING BOX")
    args = parser.parse_args()
    main()
