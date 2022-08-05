import jittor as jt
from jittor.dataset import Dataset

import os
from PIL import Image
import numpy as np

from jdet.utils.registry import DATASETS
from jdet.models.boxes.box_ops import rotated_box_to_bbox_np, poly_to_rotated_box_single
from .transforms import Compose
from jdet.utils.general import check_dir
from jdet.ops.nms_poly import iou_poly
from tqdm import tqdm
from jdet.data.devkits.voc_eval import voc_eval_dota


def sort_gt(gt):
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


@DATASETS.register_module()
class OCRDataset(Dataset):
    '''
    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 5),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 5), (optional field)
                'labels_ignore': <np.ndarray> (k, 5) (optional field)
            }
        },
        ...
    ]
    '''
    CLASSES = ['signboard']

    def __init__(self, dataset_dir=None, transforms=None, batch_size=1, num_workers=0, shuffle=False, drop_last=False, filter_empty_gt=True, filter_min_size=-1):
        super(OCRDataset, self).__init__(batch_size=batch_size,
                                         num_workers=num_workers, shuffle=shuffle, drop_last=drop_last)
        self.images_dir = os.path.abspath(
            os.path.join(dataset_dir, "imgs"))
        self.annotations_file = os.path.abspath(
            os.path.join(dataset_dir, "labels"))

        self.transforms = Compose(transforms)

        images = os.listdir(self.images_dir)
        self.images_name = [img.split('.')[0] for img in images]
        self.images = [os.path.join(self.images_dir, img) for img in images]
        self.gts = [os.path.join(self.annotations_file, name+'.txt')
                    for name in self.images_name]
        self.total_len = len(self.images_name)
        print("total_len:", self.total_len)

    def read_gt(self, idx):
        f = open(self.gts[idx], 'r')
        lines = f.readlines()
        bboxes = []
        labels = []
        for i in lines:
            bbox = i.split(',')
            if bbox[-1] == '0\n':   
                continue
            labels.append(1)
            bbox = bbox[:-1]
            bbox = poly_to_rotated_box_single(
                sort_gt([int(i) for i in bbox])).tolist()
            bboxes.append(bbox)
        if len(bboxes) == 0:
            return None
        return {'bboxes': np.array(bboxes), 'labels': np.array(labels), 'bboxes_ignore': np.array([])}

    def _read_ann_info(self, idx):
        while True:
            if self.images[idx].endswith('jpg') and self.read_gt(idx) is not None:
                break
            idx = np.random.choice(np.arange(self.total_len))
        image = Image.open(self.images[idx]).convert("RGB")

        width, height = image.size
        anno = self.read_gt(idx)

        hboxes, polys = rotated_box_to_bbox_np(anno["bboxes"])
        hboxes_ignore, polys_ignore = rotated_box_to_bbox_np(
            anno["bboxes_ignore"])

        ann = dict(
            rboxes=anno['bboxes'].astype(np.float32),
            hboxes=hboxes.astype(np.float32),
            polys=polys.astype(np.float32),
            labels=anno['labels'].astype(np.int32),
            rboxes_ignore=anno['bboxes_ignore'].astype(np.float32),
            hboxes_ignore=hboxes_ignore,
            polys_ignore=polys_ignore,
            classes=self.CLASSES,
            ori_img_size=(width, height),
            img_size=(width, height),
            scale_factor=1.0,
            filename=self.images_name[idx],
            img_file=self.images[idx])
        return image, ann

    def collate_batch(self, batch):
        imgs = []
        anns = []
        max_width = 0
        max_height = 0
        for image, ann in batch:
            height, width = image.shape[-2], image.shape[-1]
            # print(height, width)
            max_width = max(max_width, width)
            max_height = max(max_height, height)
            imgs.append(image)
            anns.append(ann)
        N = len(imgs)
        batch_imgs = np.zeros((N, 3, max_height, max_width), dtype=np.float32)
        for i, image in enumerate(imgs):
            batch_imgs[i, :, :image.shape[-2], :image.shape[-1]] = image

        return batch_imgs, anns

    def __getitem__(self, idx):
        if "BATCH_IDX" in os.environ:
            idx = int(os.environ['BATCH_IDX'])
        image, anno = self._read_ann_info(idx)

        if self.transforms is not None:
            image, anno = self.transforms(image, anno)

        return image, anno

    def evaluate(self, results, work_dir, epoch, logger=None, save=True):
        print("Calculating mAP Recall and Precision......")
        if save:
            save_path = os.path.join(work_dir, f"detections/val_{epoch}")
            check_dir(save_path)
            jt.save(results, save_path+"/val.pkl")
        dets = []
        gts = []
        diffcult_polys = {}
        for img_idx, (result, target) in enumerate(results):
            det_polys, det_scores, det_labels = result
            det_labels += 1
            if det_polys.size > 0:
                idx1 = np.ones((det_labels.shape[0], 1))*img_idx
                det = np.concatenate(
                    [idx1, det_polys, det_scores.reshape(-1, 1), det_labels.reshape(-1, 1)], axis=1)
                dets.append(det)

            scale_factor = target["scale_factor"]
            gt_polys = target["polys"]
            gt_polys /= scale_factor

            if gt_polys.size > 0:
                gt_labels = target["labels"].reshape(-1, 1)
                idx2 = np.ones((gt_labels.shape[0], 1))*img_idx
                gt = np.concatenate([idx2, gt_polys, gt_labels], axis=1)
                gts.append(gt)
            diffcult_polys[img_idx] = target["polys_ignore"]/scale_factor
        if len(dets) == 0:
            aps = {}
            for i, classname in tqdm(enumerate(self.CLASSES), total=len(self.CLASSES)):
                aps["eval/"+str(i+1)+"_"+classname+"_AP"] = 0
            map = sum(list(aps.values()))/len(aps)
            aps["eval/0_meanAP"] = map
            return aps
        dets = np.concatenate(dets)
        gts = np.concatenate(gts)
        aps = {}
        for i, classname in tqdm(enumerate(self.CLASSES), total=len(self.CLASSES)):
            c_dets = dets[dets[:, -1] == (i+1)][:, :-1]
            c_gts = gts[gts[:, -1] == (i+1)][:, :-1]
            img_idx = gts[:, 0].copy()
            classname_gts = {}
            for idx in np.unique(img_idx):
                g = c_gts[c_gts[:, 0] == idx, :][:, 1:]
                dg = diffcult_polys[idx].copy().reshape(-1, 8)
                diffculty = np.zeros(g.shape[0]+dg.shape[0])
                diffculty[int(g.shape[0]):] = 1
                diffculty = diffculty.astype(bool)
                g = np.concatenate([g, dg])
                classname_gts[idx] = {"box": g.copy(), "det": [False for i in range(
                    len(g))], 'difficult': diffculty.copy()}
            rec, prec, ap = voc_eval_dota(
                c_dets, classname_gts, iou_func=iou_poly)
            aps["eval/"+str(i+1)+"_"+classname+"_AP"] = ap
        map = sum(list(aps.values()))/len(aps)
        aps["eval/0_meanAP"] = map
        aps["eval/rec"] = rec[-1]
        aps["eval/prec"] = prec[-1]
        return aps
