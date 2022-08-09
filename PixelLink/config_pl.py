import os

curr_path = os.path.dirname(__file__)


class Config(object):
    def __init__(self) -> None:
        self.version = "2s"
        self.epoch = 100  # 500
        self.schedule = [100, 200]
        self.save_freq = 5  # 20
        self.vis_freq = 50  # 20
        self.val_freq = 10  # 20
        self.lr = 1e-3  # 1e-3
        self.weight_decay = 5e-4
        self.batch_size = 8  # 8
        self.num_workers = 1  # 8
        self.momentum = 0.9

        self.pixel_weight = 2
        self.link_weight = 1

        self.input_size = (512, 512)

        self.link_weight = 1
        self.pixel_weight = 2
        self.neg_pos_ratio = 3  # parameter r in paper

        self.dataset_path = "/home/data"
        self.images_dir = os.path.join(self.dataset_path, '1306')
        self.labels_dir = os.path.join(self.dataset_path, '1306')

        self.save_dir = os.path.join(curr_path, "work_dirs/")


cfg = Config()
