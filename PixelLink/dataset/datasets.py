import numpy as np
from PIL import Image
from jittor.dataset import Dataset
import os
import codecs
import cv2
import jittor as jt


class TencentDataLoader(Dataset):
    def __init__(self,
                 cfg,
                 mode='train',
                 transform=None,
                 batch_size=16,
                 shuffle=False,
                 drop_last=True,
                 num_workers=0):
        if mode != 'train':
            super().__init__(batch_size=1,
                             shuffle=False,
                             drop_last=False,
                             num_workers=1)
        else:
            super().__init__(batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=drop_last,
                             num_workers=num_workers)
        self.cfg = cfg
        self.mode = mode
        self.transform = transform
        self.img_paths = []
        self.gts = []

        img_dir = self.cfg.images_dir
        gt_dir = self.cfg.labels_dir

        # img_ids = self.get_ids()[0:50]  # test small number of data
        img_ids = self.get_ids()

        for img_id in img_ids:
            img_path = os.path.join(img_dir, img_id + ".jpg")
            self.img_paths.append(img_path)
            gt_path = os.path.join(gt_dir, img_id + '.txt')
            self.gts.append(self.get_bboxes(gt_path))

    def get_ids(self):
        labels_files = os.listdir(self.cfg.labels_dir)
        ids = [x.rstrip().split(".")[0] for x in labels_files if ".txt" in x]
        if self.mode == "train":
            ids = ids
        elif self.mode == "val":
            ids = ids[-100:]    # Todo
        else:
            raise NotImplementedError("Not valid mode: {}".format(self.mode))
        return ids

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = np.array(Image.open(img_path))  # rgb

        if self.mode == 'train' or self.mode == 'val':
            gt = self.gts[index].copy()
            image, gt = self.transform(image, gt)
            pixel_mask, neg_pixel_mask, pixel_pos_weight, link_mask = self.label_to_mask_and_pixel_pos_weight(
                gt, list(image.shape[1:]))
        else:
            image, _ = self.transform(image, None)
            return image

        return image, pixel_mask, neg_pixel_mask, gt, pixel_pos_weight, link_mask, img_path

    def __len__(self):
        return len(self.img_paths)

    def get_bboxes(self, gt_path):
        with codecs.open(gt_path, encoding="utf-8_sig") as file:
            data = file.readlines()
            tmp = {}
            tmp["bboxes"] = []
            tmp["content"] = []
            tmp["ignore"] = []
            tmp["area"] = []
            for line in data:
                content = line.split(",")
                coor = [int(float(n)) for n in content[:8]]
                tmp["bboxes"].append(coor)
                content[8] = content[8].strip("\r\n")
                tmp["content"].append(content[8])
                if content[8] == "###":
                    tmp["ignore"].append(True)
                else:
                    tmp["ignore"].append(False)
                coor = np.array(coor).reshape([4, 2])
                tmp["area"].append(cv2.contourArea(coor))
        return tmp

    def label_to_mask_and_pixel_pos_weight(self, label, img_size, version="2s", neighbors=8):
        """
        8 neighbors:
            0 1 2
            7 - 3
            6 5 4
        """
        factor = 2 if version == "2s" else 4
        ignore = label["ignore"]
        label = label["bboxes"]
        assert len(ignore) == len(label)
        label = np.array(label)
        label = label.reshape([-1, 1, 4, 2])
        pixel_mask_size = [int(i / factor) for i in img_size]
        link_mask_size = [neighbors, ] + pixel_mask_size

        pixel_mask = np.zeros(pixel_mask_size, dtype=np.uint8)
        pixel_weight = np.zeros(pixel_mask_size, dtype=np.float)
        link_mask = np.zeros(link_mask_size, dtype=np.uint8)
        label = (label / factor).astype(int)

        real_box_num = 0
        for i in range(label.shape[0]):
            if not ignore[i]:
                pixel_mask_tmp = np.zeros(pixel_mask_size, dtype=np.uint8)
                cv2.drawContours(pixel_mask_tmp, label[i], -1, 1, thickness=-1)
                pixel_mask += pixel_mask_tmp
        neg_pixel_mask = (pixel_mask == 0).astype(np.uint8)
        pixel_mask[pixel_mask != 1] = 0
        # assert not (pixel_mask>1).any()
        pixel_mask_area = np.count_nonzero(pixel_mask)  # total area

        for i in range(label.shape[0]):
            if not ignore[i]:
                pixel_mask_tmp = np.zeros(pixel_mask_size, dtype=np.uint8)
                cv2.drawContours(pixel_mask_tmp, label[i], -1, 1, thickness=-1)
                pixel_mask_tmp *= pixel_mask
                if np.count_nonzero(pixel_mask_tmp) > 0:
                    real_box_num += 1
        if real_box_num == 0:
            return jt.int64(pixel_mask), jt.int64(neg_pixel_mask), jt.Var(pixel_weight), jt.int64(link_mask)
        avg_weight_per_box = pixel_mask_area / real_box_num

        for i in range(label.shape[0]):  # num of box
            if not ignore[i]:
                pixel_weight_tmp = np.zeros(pixel_mask_size, dtype=np.float)
                cv2.drawContours(pixel_weight_tmp, [
                                 label[i]], -1, avg_weight_per_box, thickness=-1)
                pixel_weight_tmp *= pixel_mask
                area = np.count_nonzero(pixel_weight_tmp)  # area per box
                if area <= 0:
                    continue
                pixel_weight_tmp /= area
                pixel_weight += pixel_weight_tmp

                weight_tmp_nonzero = pixel_weight_tmp.nonzero()
                link_mask_tmp = np.zeros(pixel_mask_size, dtype=np.uint8)
                link_mask_tmp[weight_tmp_nonzero] = 1
                link_mask_shift = np.zeros(link_mask_size, dtype=np.uint8)
                w_index = weight_tmp_nonzero[1]
                h_index = weight_tmp_nonzero[0]
                w_index1 = np.clip(w_index + 1, a_min=None,
                                   a_max=link_mask_size[1] - 1)
                w_index_1 = np.clip(w_index - 1, a_min=0, a_max=None)
                h_index1 = np.clip(h_index + 1, a_min=None,
                                   a_max=link_mask_size[2] - 1)
                h_index_1 = np.clip(h_index - 1, a_min=0, a_max=None)
                link_mask_shift[0][h_index1, w_index1] = 1
                link_mask_shift[1][h_index1, w_index] = 1
                link_mask_shift[2][h_index1, w_index_1] = 1
                link_mask_shift[3][h_index, w_index_1] = 1
                link_mask_shift[4][h_index_1, w_index_1] = 1
                link_mask_shift[5][h_index_1, w_index] = 1
                link_mask_shift[6][h_index_1, w_index1] = 1
                link_mask_shift[7][h_index, w_index1] = 1

                for j in range(neighbors):
                    # +0 to convert bool array to int array
                    link_mask[j] += np.logical_and(link_mask_tmp,
                                                   link_mask_shift[j]).astype(np.uint8)
        return jt.int64(pixel_mask), jt.int64(neg_pixel_mask), jt.Var(pixel_weight), jt.int64(link_mask)

    def collate_batch(self, batch):
        images = []
        pixel_masks = []
        neg_pixel_masks = []
        pixel_pos_weights = []
        link_masks = []
        img_paths = []
        gts = {}
        gts['bboxes'] = []
        gts["content"] = []
        gts["ignore"] = []
        gts["area"] = []
        for image, pixel_mask, neg_pixel_mask, gt, pixel_pos_weight, link_mask, img_path in batch:
            images.append(jt.Var(image))
            pixel_masks.append(pixel_mask)
            neg_pixel_masks.append(neg_pixel_mask)
            pixel_pos_weights.append(pixel_pos_weight)
            link_masks.append(link_mask)
            img_paths.append(img_path)
            gts['bboxes'].append(gt['bboxes'])
            gts["content"].append(gt["content"])
            gts["ignore"].append(gt["ignore"])
            gts["area"].append(gt["area"])
        images = jt.stack(images, dim=0)
        pixel_masks = jt.stack(pixel_masks, dim=0)
        neg_pixel_masks = jt.stack(neg_pixel_masks, dim=0)
        pixel_pos_weights = jt.stack(pixel_pos_weights, dim=0)
        link_masks = jt.stack(link_masks, dim=0)

        return images, pixel_masks, neg_pixel_masks, gts, pixel_pos_weights, link_masks, img_paths
