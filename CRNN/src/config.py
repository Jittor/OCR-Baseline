import os

curr_path = os.path.dirname(__file__)

rnn_hidden = 256

img_channel = 1
img_width = 256  # 200  # 100
img_height = 64  # 64  # 32

epoch = 200

train_batch_size = 32  # 32
valid_batch_size = 1

learning_rate = 5e-5  # 5e-4

pred_threshold = 0.6

valid_interval = 1000  # 2000
save_interval = valid_interval * 2  # 8000

datasets_path = os.path.join("/home/gmh/dataset/dataset/dataset")

images_dir = os.path.join(datasets_path, 'dataset3', 'imgs')
labels_dir = os.path.join(datasets_path, 'dataset3', 'labels')

# pretrain dataset - Synthetic Chinese String Dataset
pretrain_images_dir = "/mnt/disk/gmh/dataset/chinese/images/"
pretrain_train_text = "/mnt/disk/gmh/dataset/chinese/train.txt"
pretrain_test_text = "/mnt/disk/gmh/dataset/chinese/test.txt"

# pretrain dataset - ICDAR2019-LSVT
ICDAR2019_images_dir = "/mnt/disk/llt/data/ICDAR2019/train_full_images"
ICDAR2019_labels_file = "/mnt/disk/llt/data/ICDAR2019/train_full_labels.json"
ICDAR2019WEAK_images_root = "/mnt/disk/llt/data/ICDAR2019/"
ICDAR2019WEAK_labels_file = "/mnt/disk/llt/data/ICDAR2019/train_weak_labels.json"

lexicons_path = os.path.join(curr_path, "../lexicon/")

alphabet_path = os.path.join(curr_path, r"../data/char_std_5990.txt")     # Use standard char

save_dir = os.path.join(curr_path, "../checkpoints/")

output_file = os.path.join(curr_path, "../evaluation.txt")
