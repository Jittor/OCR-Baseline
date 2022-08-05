import jittor as jt
from jittor import nn
import cv2
import numpy as np
from config_pl import cfg


def cal_label_on_batch(out_1, out_2):
    scale = 2 if cfg.version == "2s" else 4
    all_boxes = mask_to_box(out_1, out_2, scale=scale)
    return all_boxes


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


def comp_gt_and_output(my_labels, gt_labels, threshold):
    """
    return: [true_pos, false_pos, false_neg]
    """
    coor = gt_labels["bboxes"][0]
    ignore = gt_labels["ignore"][0]
    my_labels = my_labels[0]
    true_pos, true_neg, false_pos, false_neg = [0] * 4
    for my_label in my_labels:
        for gt_label in coor:
            if cal_iou(my_label, gt_label) > threshold:
                true_pos += 1
                break
        else:
            false_pos += 1
    for i, gt_label in enumerate(coor):
        if ignore[i]:
            continue
        for my_label in my_labels:
            if cal_iou(gt_label, my_label) > threshold:
                break
        else:
            false_neg += 1
    return true_pos, false_pos, false_neg


def mask_to_box(pixel_mask, link_mask, neighbors=8, scale=4):
    """
    pixel_mask: batch_size * 2 * H * W
    link_mask: batch_size * 16 * H * W
    """
    def distance(a, b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    def short_side_filter(bounding_box):
        for i, point in enumerate(bounding_box):
            if distance(point, bounding_box[(i+1) % 4]) < 5**2:
                return True  # ignore it
        return False  # do not ignore
    batch_size = link_mask.size(0)
    mask_height = link_mask.size(2)
    mask_width = link_mask.size(3)
    # TODO
    pixel_class = nn.softmax(pixel_mask, dim=-3)
    pixel_class = pixel_class[:, 1] > 0.7
    link_neighbors = jt.zeros(
        (batch_size, neighbors, mask_height, mask_width), dtype=jt.uint8)

    for i in range(neighbors):
        tmp = nn.softmax(link_mask[:, [2 * i, 2 * i + 1]], dim=-3)
        link_neighbors[:, i] = tmp[:, 1] > 0.7
        link_neighbors[:, i] = link_neighbors[:, i] & pixel_class
    all_boxes = []
    for i in range(batch_size):
        res_mask = func(pixel_class[i], link_neighbors[i])
        box_num = np.max(res_mask)
        bounding_boxes = []
        for j in range(1, box_num + 1):
            box_mask = (res_mask == j).astype(np.uint8)
            if box_mask.sum() < 100:
                continue

            # box_mask, contours, _ = cv2.findContours(
            contours, _ = cv2.findContours(
                box_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

            bounding_box = cv2.minAreaRect(contours[0])
            bounding_box = cv2.boxPoints(bounding_box)
            if short_side_filter(bounding_box):
                continue

            bounding_box = np.clip(bounding_box * scale,
                                   # 0, 128 * scale - 1).astype(np.int)
                                   0, 256 * scale - 1).astype(np.int)

            bounding_boxes.append(bounding_box)
        all_boxes.append(bounding_boxes)
    return all_boxes


def get_neighbors(h_index, w_index):
    res = []
    res.append((h_index - 1, w_index - 1))
    res.append((h_index - 1, w_index))
    res.append((h_index - 1, w_index + 1))
    res.append((h_index, w_index + 1))
    res.append((h_index + 1, w_index + 1))
    res.append((h_index + 1, w_index))
    res.append((h_index + 1, w_index - 1))
    res.append((h_index, w_index - 1))
    return res


def func(pixel_cls, link_cls):
    def joint(pointa, pointb):
        roota = find_root(pointa)
        rootb = find_root(pointb)
        if roota != rootb:
            group_mask[rootb] = roota
        return

    def find_root(pointa):
        root = pointa
        while group_mask.get(root) != -1:
            root = group_mask.get(root)
        return root

    pixel_cls = pixel_cls.numpy()
    link_cls = link_cls.numpy()

    pixel_points = list(zip(*np.where(pixel_cls)))
    h, w = pixel_cls.shape
    group_mask = dict.fromkeys(pixel_points, -1)

    for point in pixel_points:
        h_index, w_index = point

        neighbors = get_neighbors(h_index, w_index)
        for i, neighbor in enumerate(neighbors):
            nh_index, nw_index = neighbor
            if nh_index < 0 or nw_index < 0 or nh_index >= h or nw_index >= w:
                continue
            if pixel_cls[nh_index, nw_index] == 1 and link_cls[i, h_index, w_index] == 1:
                joint(point, neighbor)

    res = np.zeros(pixel_cls.shape, dtype=np.uint8)
    root_map = {}
    for point in pixel_points:
        h_index, w_index = point
        root = find_root(point)
        if root not in root_map:
            root_map[root] = len(root_map) + 1
        res[h_index, w_index] = root_map[root]

    return res
