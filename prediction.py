from CRNN.step3 import Step3
from PixelLink.step2 import Step2
from argparse import ArgumentParser
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import sys
from JDet.step1 import Step1

sys.path.extend(['PixelLink', 'CRNN', 'CRNN/src'])


def sort_point(gt):
    def take_first(elem):
        return elem[0]

    gt_ = [[gt[i], gt[i + 1]] for i in [0, 2, 4, 6]]
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
    w = max(gt[2] - gt[0], gt[4] - gt[6], gt[2] - gt[6], gt[4] - gt[0])
    h = max(gt[7] - gt[1], gt[5] - gt[3], gt[7] - gt[3], gt[5] - gt[1])
    return max(int(x), 0), max(int(y), 0), int(w), int(h)


def crop_img(bounding_box, img):
    iw, ih = img.size
    x_2, y_2, w_2, h_2 = get_xywh(bounding_box)
    if x_2 > iw or y_2 > ih or w_2 == 0 or h_2 == 0:
        return None
    new_img = Image.fromarray(
        np.array(img)[y_2:min(y_2 + h_2, ih), x_2:min(x_2 + w_2, iw)])
    return new_img


def get_max_img_id(img_list):
    img_area = np.array([image.size[0] * image.size[1] for image in img_list])
    return np.argmax(img_area)


def output_result(texts, signboard_boxes, output_text_path):
    assert len(texts) == len(signboard_boxes)
    lines = list()
    for text, signboard_box in zip(texts, signboard_boxes):
        if not text:
            continue
        signboard_box = [str(int(point)) for point in signboard_box]
        signboard_box.append(text)
        lines.append(",".join(signboard_box))
    lines = "\n".join(lines)
    train_file = open(output_text_path, "w", encoding='utf-8')
    train_file.write(lines)
    train_file.close()
    # print(lines)


def pipeline(input_img_path, output_text_path, model1, model2, model3):
    img = Image.open(input_img_path).convert("RGB")
    # if "017049_1508415750706032.jpg" not in input_img_path:
    #     return

    signboard_bbox_list = list()
    text_list = list()

    # Step 1 JDet
    signboard_bboxes = model1.infer(img)
    signboards = list()
    for i, signboard_bbox in enumerate(signboard_bboxes):
        signboard_bbox = sort_point(signboard_bbox.tolist())
        signboard_img = crop_img(signboard_bbox, img)
        if not signboard_img:
            continue
        # signboard.save("./test_signboard_" + str(i) + ".jpg")  # check signboard img
        signboards.append(signboard_img)
        signboard_bbox_list.append(signboard_bbox)
    if not signboard_bbox_list:
        output_result(str(), list(), output_text_path)
        return

    # Step 2 PixelLink
    max_text_box_list = list()
    for signboard in signboards:
        text_bboxes = model2.infer(signboard)
        text_boxes = list()
        for j, text_bbox in enumerate(text_bboxes):
            text_bbox = text_bbox.reshape(-1).tolist()
            text_box_img = crop_img(text_bbox, signboard)
            if not text_box_img:
                continue
            # text_box.save("./test_text_box_" + str(j) + ".jpg")  # check text_box img
            text_boxes.append(text_box_img)
        if not text_boxes:
            max_text_box_list.append(list())
        else:
            max_box_id = get_max_img_id(text_boxes)
            max_text_box_list.append(text_boxes[max_box_id])

    # Step 3 CRNN
    for text_box in max_text_box_list:
        if text_box:
            text = model3.infer(text_box)
            text_list.append(text)
        else:
            text_list.append(str())

    # Output the prediction
    output_result(text_list, signboard_bbox_list, output_text_path)


def main():
    # Load models
    model1 = Step1('JDet/store_sign_detection/s2anet_r50_fpn_5x_ocr_630_1120_bs4.py',
                   args.jdet)
    model2 = Step2(args.pixellink)
    model3 = Step3(args.crnn)

    # Pipeline prediction
    os.makedirs(args.save_dir, exist_ok=True)
    img_names = os.listdir(args.image_dir)
    pbar = tqdm(enumerate(img_names), total=len(img_names), desc="Test")
    for i, img_name in pbar:
        pbar.set_description("Test - i:{}, {}".format(i, img_name))
        input_img_path = os.path.join(args.image_dir, img_name)
        img_id = img_name.split(".")[0]
        output_text_path = args.save_dir + img_id + ".txt"
        pipeline(input_img_path, output_text_path, model1, model2, model3)
    pbar.close()
    print("Please find the output files in %s" % args.save_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i",
                        "--image_dir",
                        default="dataset/test_dataset/imgs",
                        type=str,
                        help="path of image files.",
                        metavar="IMAGES PATH")
    parser.add_argument("-s",
                        "--save_dir",
                        default="./prediction/",
                        type=str,
                        help="path to save prediction files.",
                        metavar="SAVE PATH")
    parser.add_argument("-j",
                        "--jdet",
                        default="./ckpts/JDet.pkl",
                        type=str,
                        help="path of JDet model.",
                        metavar="JDET MODEL")
    parser.add_argument("-p",
                        "--pixellink",
                        default="./ckpts/PixelLink.pkl",
                        type=str,
                        help="path of PixelLink model.",
                        metavar="PIXELLINK MODEL")
    parser.add_argument("-c",
                        "--crnn",
                        default="./ckpts/CRNN.pkl",
                        type=str,
                        help="path of CRNN model",
                        metavar="CRNN MODEL")
    args = parser.parse_args()
    main()


# Commands
# python prediction.py -i [街景图片目录] -s [保存预测文件目录] -j [JDet模型参数路径] -p [PixelLink模型参数路径] -c [CRNN模型参数路径]
