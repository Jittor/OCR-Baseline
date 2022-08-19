from CRNN.step3 import Step3
from PixelLink.step2 import Step2
from JDet.step1 import Step1
from PIL import Image
import numpy as np
import cv2
import sys
import os

curr_path = os.path.dirname(__file__)
sys.path.extend([
    os.path.join(curr_path, './'),
    os.path.join(curr_path, './PixelLink'),
    os.path.join(curr_path, './CRNN'),
    os.path.join(curr_path, './CRNN/src'),
])

# Setting
JDet_PATH = "/project/train/models/JDet.pkl"
PixelLink_PATH = "/project/train/models/PixelLink.pkl"
CRNN_PATH = "/project/train/models/CRNN.pkl"
JDet_CFG_PATH = './JDet/store_sign_detection/s2anet_r50_fpn_5x_ocr_630_1120_bs4.py'


class Pipeline:
    def __init__(self,
                 jdet_path=os.path.join(curr_path, JDet_PATH),
                 pixellink_path=os.path.join(curr_path, PixelLink_PATH),
                 crnn_path=os.path.join(curr_path, CRNN_PATH),
                 ):
        # Load models
        self.model1 = Step1(os.path.join(curr_path, JDet_CFG_PATH), jdet_path)
        self.model2 = Step2(pixellink_path)
        self.model3 = Step3(crnn_path)

    def infer(self, image):
        image = image[..., [2, 1, 0]]  # cv2.imread input
        image = Image.fromarray(image)
        signboard_bbox_list = list()
        text_list = list()

        # Step 1 JDet
        signboard_bboxes = self.model1.infer(image)
        signboards = list()
        for i, signboard_bbox in enumerate(signboard_bboxes):
            signboard_bbox = sort_point(signboard_bbox.tolist())
            signboard_img = crop_img(signboard_bbox, image)
            if not signboard_img:
                continue
            # signboard.save("./test_signboard_" + str(i) + ".jpg")  # check signboard img
            signboards.append(signboard_img)
            signboard_bbox_list.append(signboard_bbox)
        if not signboard_bbox_list:
            return output_result(str(), list())

        # Step 2 PixelLink
        max_text_box_list = list()
        for signboard in signboards:
            text_bboxes = self.model2.infer(signboard)
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
                text = self.model3.infer(text_box)
                text_list.append(text)
            else:
                text_list.append(str())

        # Output the prediction
        return output_result(text_list, signboard_bbox_list)

    def __call__(self, *args, **kwargs):
        return self.infer(*args, **kwargs)


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


def output_result(texts, signboard_boxes):
    assert len(texts) == len(signboard_boxes)
    lines = list()
    for text, signboard_box in zip(texts, signboard_boxes):
        if not text:
            continue
        signboard_box = [(int(point)) for point in signboard_box]
        signboard_box.append(text)
        lines.append(signboard_box)
    return lines


if __name__ == "__main__":
    pipeline = Pipeline()
    img = cv2.imread('/home/data/1413/011284_1508290961970980.jpg')
    results = pipeline(img)
    print(results)
