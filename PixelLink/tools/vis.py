import numpy as np
import cv2


def draw_labels(img, labels, data_format="HWC", color_format="RGB", contour_color=(0, 255, 0), width=1):
    labels = np.array(labels).reshape([-1, 4, 2])
    if color_format == "RGB":
        img = img[..., [2, 1, 0]]
    if data_format == "CHW":
        img = img.transpose(1, 2, 0)
    img = np.ascontiguousarray(img)
    cv2.drawContours(img, labels, contourIdx=-1,
                     color=contour_color, thickness=width)
    return img


def val_draw_labels(image, gt, epoch, iter, predict=None, img_name=None):
    # print("predict:", predict)
    bboxes = gt['bboxes']
    bboxes = np.array(bboxes[0])
    ignore = gt['ignore'][0]
    ignore = [not i for i in ignore]
    bboxes = bboxes[ignore]
    predict = predict[0]
    image = image[0]
    image = image.transpose(1, 2, 0).numpy()
    image = (image*np.float32([0.229, 0.224, 0.225]).reshape(1, 1, -1) +
             np.float32([0.485, 0.456, 0.406]).reshape(1, 1, -1))*255
    if predict is not None:
        image = draw_labels(image, predict, contour_color=(0, 255, 0))  # green

    image = draw_labels(image,
                        bboxes, contour_color=(0, 0, 255))  # red

    cv2.imwrite("work_dirs/{}_{}-{}.jpg".format(epoch, iter, img_name), image)
