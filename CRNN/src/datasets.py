import os
import json
from PIL import Image

import jittor as jt
from jittor.dataset import Dataset

from utils import get_alphabet, get_xywh, process_img, valid_gt_text
import config

CHARS = get_alphabet(path=config.alphabet_path)
# CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}


class CommonDataset(Dataset):
    def __init__(self,
                 batch_size=16,
                 shuffle=False,
                 drop_last=False,
                 num_workers=0,
                 buffer_size=536870912,
                 stop_grad=True,
                 keep_numpy_array=False,
                 endless=False):
        super().__init__(batch_size=batch_size,
                         shuffle=shuffle,
                         drop_last=drop_last,
                         num_workers=num_workers,
                         buffer_size=buffer_size,
                         stop_grad=stop_grad,
                         keep_numpy_array=keep_numpy_array,
                         endless=endless)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        try:
            if config.img_channel == 1:
                ori_image = Image.open(img_path).convert('L')  # grey-scale
            elif config.img_channel == 3:
                ori_image = Image.open(img_path)               # rgb
            else:
                raise Exception("Please set img_channel 1 or 3.")
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        num_sub_img = len(self.texts[index])
        images = list()
        targets = list()
        target_lengths = list()
        for i in range(num_sub_img):
            text, box = self.texts[index][i], self.boxes[index][i]

            if box == -1:
                img_crop = ori_image
            elif len(box) == 8:
                box = [int(float(i)) for i in box]
                x, y, w, h = get_xywh(box)
                img_crop = ori_image.crop((x, y, x + w, y + h))
            else:
                raise Exception("Not valid bounding box: %s\nPlease check img id: %s" % (box, self.img_ids[index]))
            # img_crop.save('{}.jpg'.format(text))
            # print("text", text)
            # raise

            img_crop = process_img(img_crop, img_width=self.img_width, img_height=self.img_height, img_channel=config.img_channel)
            images.append(img_crop.unsqueeze(0))

            target = [CHAR2LABEL[c] for c in text if c in CHARS]
            target = jt.int32(target)
            targets.append(target)

            target_length = len(target)
            target_lengths.append(target_length)

        images = jt.concat(images, 0)
        target_lengths = jt.int32(target_lengths)

        max_target_length = target_lengths.max()
        targets = [t.reindex([max_target_length.item()], ["i0"]) for t in targets]
        targets = jt.stack(targets, dim=0)

        img_ids = [self.img_ids[index]] * len(images)
        return images, targets, target_lengths, img_ids

    def collate_batch(self, batch):
        images, targets, target_lengths, img_ids = zip(*batch)
        images = jt.concat(images, dim=0)

        target_lengths = jt.concat(target_lengths, dim=0)

        max_target_length = target_lengths.max()
        new_target_list = list()
        for target in targets:
            target = [t.reindex([max_target_length.item()], ["i0"]) for t in target]
            new_target_list += target
        targets = jt.stack(new_target_list)

        img_ids = sum(img_ids, list())

        return images, targets, target_lengths, img_ids


class TencentData(CommonDataset):
    def __init__(self,
                 mode,
                 root_dir=None,
                 images_dir=config.images_dir,
                 labels_dir=config.labels_dir,
                 img_height=32,
                 img_width=100,
                 batch_size=16,
                 shuffle=False,
                 drop_last=False,
                 num_workers=0,
                 buffer_size=536870912,
                 stop_grad=True,
                 keep_numpy_array=False,
                 endless=False):
        super().__init__(batch_size=batch_size,
                         shuffle=shuffle,
                         drop_last=drop_last,
                         num_workers=num_workers,
                         buffer_size=buffer_size,
                         stop_grad=stop_grad,
                         keep_numpy_array=keep_numpy_array,
                         endless=endless)
        self.mode = mode
        self.labels_dir = labels_dir
        self.img_height = img_height
        self.img_width = img_width
        self.img_paths = []
        self.img_ids = []
        self.texts = []
        self.boxes = []

        # img_ids = self.get_ids()[:50]  # test small number of data
        img_ids = self.get_ids()

        for img_id in img_ids:
            img_path = os.path.join(images_dir, img_id + ".jpg")
            label_path = os.path.join(labels_dir, img_id + '.txt')

            texts, boxes = self.get_text_and_box(label_path)
            if not boxes:
                continue

            self.img_paths.append(img_path)
            self.img_ids.append(img_id)
            self.texts.append(texts)
            self.boxes.append(boxes)

    def get_ids(self):
        labels_files = os.listdir(self.labels_dir)
        ids = [x.rstrip().split(".")[0] for x in labels_files]
        if self.mode == "train":
            ids = ids
        elif self.mode == "val":
            ids = ids[-100:]    # Todo
        else:
            raise NotImplementedError("Not valid mode: {}".format(self.mode))
        return ids

    def get_text_and_box(self, path):
        texts = list()
        boxes = list()
        with open(path, encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            text = line.rstrip().split(",")[-1]
            box = line.rstrip().split(",")[:8]
            if len(text) > config.img_width / 4:
                continue  # too long text
            if self.mode == "val" and not valid_gt_text(text):
                continue
            texts.append(text)
            boxes.append(box)
        return texts, boxes


class ICDAR2019(CommonDataset):
    def __init__(self,
                 mode,
                 root_dir=None,
                 images_dir=config.ICDAR2019_images_dir,
                 labels_file=config.ICDAR2019_labels_file,
                 img_height=32,
                 img_width=100,
                 batch_size=16,
                 shuffle=False,
                 drop_last=False,
                 num_workers=0,
                 # buffer_size=536870912,
                 stop_grad=True,
                 keep_numpy_array=False,
                 endless=False):
        super().__init__(batch_size=batch_size,
                         shuffle=shuffle,
                         drop_last=drop_last,
                         num_workers=num_workers,
                         # buffer_size=buffer_size,
                         stop_grad=stop_grad,
                         keep_numpy_array=keep_numpy_array,
                         endless=endless)
        self.mode = mode
        self.root_dir = root_dir
        self.images_dir = images_dir
        self.labels_file = labels_file
        self.img_height = img_height
        self.img_width = img_width
        self.img_paths = []
        self.img_ids = []
        self.texts = []
        self.boxes = []

        self.labels = self.get_labels()
        # img_ids = list(self.labels.keys())[:50]  # test small number of data
        img_ids = list(self.labels.keys())
        # split_point = int(len(img_ids)*0.95)
        img_ids = img_ids[:-2000] if self.mode == "train" else img_ids[-2000:]

        for img_id in img_ids:
            img_path = self.get_img_path(img_id)

            texts, boxes = self.get_text_and_box(img_id)
            if not boxes:
                continue

            self.img_paths.append(img_path)
            self.img_ids.append(img_id)
            self.texts.append(texts)
            self.boxes.append(boxes)

    def get_labels(self):
        with open(self.labels_file, encoding='utf-8') as f:
            labels = json.load(f)
        return labels

    def get_text_and_box(self, img_id):
        labels = self.labels.get(img_id)
        texts = list()
        boxes = list()
        for label in labels:
            if label["illegibility"]:
                continue
            text = label["transcription"]
            box = sum(label["points"], [])
            if len(text) > config.img_width / 4:
                continue  # too long text
            if len(box) != 8:
                continue
            texts.append(text)
            boxes.append(box)
        return texts, boxes

    def get_img_path(self, img_id):
        img_path = os.path.join(self.images_dir, img_id + ".jpg")
        return img_path


class ICDAR2019WEAK(ICDAR2019):
    def __init__(self,
                 root_dir=None,
                 labels_file=config.ICDAR2019WEAK_labels_file,
                 *inputs, **kwargs):
        if not root_dir:
            root_dir = config.ICDAR2019WEAK_images_root
        super().__init__(root_dir=root_dir, labels_file=labels_file, *inputs, **kwargs)

    def get_text_and_box(self, img_id):
        labels = self.labels.get(img_id)
        texts = list()
        boxes = list()
        for label in labels:
            text = label["transcription"]
            box = -1
            if len(text) > config.img_width / 4:
                continue  # too long text
            texts.append(text)
            boxes.append(box)
        return texts, boxes

    def get_img_path(self, img_id):
        id_num = int(img_id.split("_")[-1])
        img_dir = "train_weak_images_{}".format(int(id_num / 40000))
        img_path = os.path.join(self.root_dir, img_dir, img_id + ".jpg")
        return img_path


class PredictDataset(Dataset):
    def __init__(self,
                 img_paths,
                 img_height=32,
                 img_width=100,
                 batch_size=16,
                 shuffle=False,
                 drop_last=False,
                 num_workers=0,
                 buffer_size=512 * 1024 * 1024,
                 stop_grad=True,
                 keep_numpy_array=False,
                 endless=False):
        super().__init__(batch_size=batch_size,
                         shuffle=shuffle,
                         drop_last=drop_last,
                         num_workers=num_workers,
                         buffer_size=buffer_size,
                         stop_grad=stop_grad,
                         keep_numpy_array=keep_numpy_array,
                         endless=endless)
        self.img_paths = img_paths
        self.total_len = len(self.img_paths)
        self.img_height = img_height
        self.img_width = img_width

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        try:
            image = Image.open(img_path).convert('L')  # grey-scale
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        image = process_img(image, img_width=self.img_width, img_height=self.img_height, img_channel=config.img_channel)
        return image


class PreTrainedDataset(Dataset):
    def __init__(self,
                 mode,
                 root_dir=None,
                 img_height=32,
                 img_width=100,
                 batch_size=16,
                 shuffle=False,
                 drop_last=False,
                 num_workers=0,
                 buffer_size=536870912,
                 stop_grad=True,
                 keep_numpy_array=False,
                 endless=False):
        super().__init__(batch_size=batch_size,
                         shuffle=shuffle,
                         drop_last=drop_last,
                         num_workers=num_workers,
                         buffer_size=buffer_size,
                         stop_grad=stop_grad,
                         keep_numpy_array=keep_numpy_array,
                         endless=endless)
        self.mode = mode
        self.img_height = img_height
        self.img_width = img_width
        self.img_paths = []
        self.img_ids = []
        self.texts = []

        img_dir = config.pretrain_images_dir

        # gt_labels = self.get_labels()[:50]  # test small number of data
        gt_labels = self.get_labels()

        for gt_label in gt_labels:
            img_path = os.path.join(img_dir, gt_label[0])
            try:
                char_list = [LABEL2CHAR[int(i)] for i in gt_label[1:]]
            except:
                continue
            gt_text = "".join(char_list)

            if len(gt_text) > 50:
                continue  # too long label

            self.img_paths.append(img_path)
            self.img_ids.append(gt_label[0])
            self.texts.append(gt_text)

    def __len__(self):
        return len(self.img_paths)

    def get_labels(self):
        if self.mode == "train":
            path = config.pretrain_train_text
        elif self.mode == "val":
            path = config.pretrain_test_text
        else:
            raise Exception('Please set the mode "train" or "test".')
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
        labels = [line.rstrip().split(" ") for line in lines]
        return labels

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        try:
            if config.img_channel == 1:
                image = Image.open(img_path).convert('L')  # grey-scale
            elif config.img_channel == 3:
                image = Image.open(img_path)               # rgb
            else:
                raise Exception("Please set img_channel 1 or 3.")
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        image = process_img(image, img_width=self.img_width, img_height=self.img_height, img_channel=config.img_channel)

        text = self.texts[index]
        target = [CHAR2LABEL[c] for c in text if c in CHARS]
        target_length = [len(target)]

        target = jt.int64(target)
        target_length = jt.int64(target_length)

        img_id = self.img_ids[index]

        return image, target, target_length, img_id

    def collate_batch(self, batch):
        images, targets, target_lengths, img_ids = zip(*batch)
        images = jt.stack(images, dim=0)

        target_lengths = jt.concat(target_lengths, dim=0)

        max_target_length = target_lengths.max()
        targets = [t.reindex([max_target_length.item()], ["i0"]) for t in targets]
        targets = jt.stack(targets, dim=0)

        return images, targets, target_lengths, img_ids
