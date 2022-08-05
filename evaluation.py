from argparse import ArgumentParser
import os
import re
from tqdm import tqdm
import numpy as np
import cv2


def valid_gt_text(text):
    return not (re.search(r"[\WA-Za-z]", text)) and len(text) >= 2


def get_text_and_box(path, verify=False):
    texts = list()
    boxes = list()
    with open(path, encoding='utf-8') as file:
        lines = file.readlines()
    for line in lines:
        text = line.rstrip().split(",")[-1]
        if verify and not valid_gt_text(text):
            continue
        texts.append(text)

        box = line.rstrip().split(",")[:8]
        box = [int(point) for point in box]
        boxes.append(box)
    return texts, boxes


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


def main():
    filenames = os.listdir(args.gt_dir)
    gt_paths = [os.path.join(args.gt_dir, filename) for filename in filenames]
    prediction_paths = [os.path.join(args.pred_dir, filename) for filename in filenames]

    output_file = open(args.output_file, "w", encoding='utf-8')
    output_file.write("Evaluation:\n")

    correct_num = 0
    pred_num = 0
    gt_num = 0
    pbar = tqdm(enumerate(filenames), total=len(filenames), desc="Test")
    for index, filename in pbar:
        pbar.set_description("Test - i:{}, {}".format(index, filename))

        # if filename not in "016343_1507987402501684.txt":
        #     continue
        gt_path = gt_paths[index]
        prediction_path = prediction_paths[index]
        gt_texts, gt_boxes = get_text_and_box(gt_path, verify=True)
        pred_texts, pred_boxes = get_text_and_box(prediction_path)
        gt_num += len(gt_texts)
        pred_num += len(pred_texts)

        for i in range(len(gt_texts)):
            score_text_list = list()
            score_bbox_list = list()
            is_correct_list = list()
            for j in range(len(pred_texts)):
                score_text = round(count_similarity(gt_texts[i], pred_texts[j]), 4)
                score_bbox = round(cal_iou(gt_boxes[i], pred_boxes[j]), 4)
                is_correct = score_text >= args.threshold_text and score_bbox >= args.threshold_box
                if is_correct:
                    correct_num += 1
                score_text_list.append(score_text)
                score_bbox_list.append(score_bbox)
                is_correct_list.append(is_correct)
            if score_bbox_list and max(score_bbox_list) > 0:
                matched_j = np.argmax(score_bbox_list)
            else:
                matched_j = -1

            # output the evaluation on each label
            output_text0 = "\niter: %s\nimg_id: %s" % (index, filename.split(".")[0])
            output_text1 = "gt_bbox: %s, gt_text: %s" % (gt_boxes[i], gt_texts[i])
            output_text2 = "pred_bbox: %s, pred_text: %s" % (
                (pred_boxes[matched_j], pred_texts[matched_j]) if matched_j != -1 else (list(), str()))
            output_text3 = "similarity: bbox (%s), text (%s)" % (
                (score_bbox_list[matched_j], score_text_list[matched_j]) if matched_j != -1 else (0, 0))
            output_text4 = "good prediction: %s\n" % ((is_correct_list[matched_j]) if matched_j != -1 else False)
            line = "\n".join([output_text0, output_text1, output_text2, output_text3, output_text4])
            output_file.write(line)
            # print(line)

    pbar.close()

    # output the overall result
    acc = round(correct_num / pred_num, 4)
    recall = round(correct_num / gt_num, 4)
    lines = list()
    lines.append("\n-------------------Evaluation-------------------")
    lines.append("gt_num: {}".format(gt_num))
    lines.append("pred_num: {}".format(pred_num))
    lines.append("correct_num: {}".format(correct_num))
    lines.append("acc: {}".format(acc))
    lines.append("recall: {}\n".format(recall))
    lines = "\n".join(lines)
    output_file.write(lines)
    output_file.close()
    print(lines)
    print("Please find the evaluation result in %s\n" % args.output_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p",
                        "--pred_dir",
                        default="./prediction/",
                        type=str,
                        help="path of prediction files.",
                        metavar="PREDICTION PATH")
    parser.add_argument("-g",
                        "--gt_dir",
                        default="/home/gmh/dataset/dataset/test_dataset/labels",
                        type=str,
                        help="path of label files.",
                        metavar="LABEL PATH")
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
    parser.add_argument("-o",
                        "--output_file",
                        default="./evaluation.txt",
                        type=str,
                        help="path of output evaluation file.",
                        metavar="EVALUATION PATH")
    args = parser.parse_args()
    main()
