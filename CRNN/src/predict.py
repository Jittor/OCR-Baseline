import os
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm
import jittor as jt

from datasets import PredictDataset, LABEL2CHAR, CHAR2LABEL, CHARS, TencentData
from model import CRNN
from ctc_decoder import ctc_decode
from utils import get_xywh, process_img
import config

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-r",
                        "--reload_checkpoint",
                        type=str,
                        help="the checkpoint to reload",
                        required=True,
                        metavar="CHECKPOINT")
    parser.add_argument("-s", "--batch_size", metavar="BATCH SIZE", type=int, default=1, help="batch size")
    parser.add_argument("-l",
                        "--lexicon_based",
                        action="store_true",
                        help="lexicon based method")
    parser.add_argument("--decode_method",
                        "-d",
                        default="greedy",
                        type=str,
                        choices=["greedy", "beam_search", "prefix_beam_search"],
                        help="decode method (greedy, beam_search or prefix_beam_search) [default: beam_search]",
                        metavar="DECODE METHOD")
    parser.add_argument("-b", "--beam_size", default=10, type=int, help="beam size [default: 10]", metavar="BEAM SIZE")
    parser.add_argument("-H",
                        "--img_height",
                        default=config.img_height,
                        type=int,
                        help="image height [default: 32]",
                        metavar="IMAGE HEIGHT")
    parser.add_argument("-W",
                        "--img_width",
                        default=config.img_width,
                        type=int,
                        help="image width [default: 100]",
                        metavar="IMAGE WIDTH")
    parser.add_argument("-n",
                        "--cpu_workers",
                        default=16,
                        type=int,
                        help="number of cpu workers used to load data [default: 16]",
                        metavar="CPU WORKERS")
    parser.add_argument("--cpu",
                        action="store_true",
                        help="use cpu for all computation, default to enable cuda when possible")
    parser.add_argument("--bbox_dir",
                        default=None,
                        type=str,
                        help="path of bboxes files.",
                        metavar="BBOXES PATH")
    parser.add_argument("--image_dir",
                        default=None,
                        type=str,
                        help="path of image files.",
                        metavar="IMAGES PATH")
    parser.add_argument("--images", nargs="+", type=str, help="path to images", metavar="IMAGE")

    args = parser.parse_args()


def predict(crnn, dataset, label2char, decode_method, beam_size):
    crnn.eval()
    all_preds = []
    with jt.no_grad():
        pbar = tqdm(total=len(dataset), desc="Predict")
        for data in dataset:
            log_probs = crnn(data)
            preds = ctc_decode(log_probs.numpy(), method=decode_method, beam_size=beam_size, label2char=label2char)
            all_preds += preds
            pbar.update(1)
        pbar.close()

    return all_preds


def show_result(paths, preds):
    print('\n===== result =====')
    for path, pred in zip(paths, preds):
        text = pred
        print(f'{path} > {text}')


def image_predict_with_boxes_using_dataset(ckp_path=args.reload_checkpoint, img_dir=args.image_dir, bbox_dir=args.bbox_dir):
    # Load model
    num_class = len(LABEL2CHAR) + 1
    crnn = CRNN(config.img_channel, args.img_height, args.img_width, num_class, rnn_hidden=config.rnn_hidden)
    crnn.load(ckp_path)
    crnn.eval()

    # Load dataset
    predict_dataset = TencentData(
        root_dir=None,
        mode=None,
        images_dir=img_dir,
        labels_dir=bbox_dir,
        img_height=args.img_height,
        img_width=args.img_width,
        batch_size=1,  # Only available for batch_size=1
        shuffle=False,
        num_workers=args.cpu_workers)

    # image_predict_with_boxes
    pbar = tqdm(predict_dataset, desc="Predict")
    for data in pbar:
        images, _, _, img_ids = data
        log_probs = crnn(images)
        preds = ctc_decode(log_probs.numpy(), method=args.decode_method, beam_size=args.beam_size,
                           label2char=LABEL2CHAR)
        bboxes_path = os.path.join(bbox_dir, img_ids[0] + ".txt")
        lines = open(bboxes_path, 'r').readlines()
        bboxes = [line.strip().split(',') for line in lines]

        file = open(bboxes_path, "w", encoding='utf-8')
        for bbox, pred in zip(bboxes, preds):
            prediction = "".join(pred)
            bbox = bbox[:8]
            bbox.append(prediction)
            line = ",".join(bbox) + "\n"
            file.write(line)
            pbar.set_description("Evaluate - pred_text:{}".format(prediction))
        file.close()
    pbar.close()


def image_predict_with_boxes(ckp_path=args.reload_checkpoint, img_dir=args.image_dir, bbox_dir=args.bbox_dir):
    # Load model
    num_class = len(LABEL2CHAR) + 1
    crnn = CRNN(config.img_channel, args.img_height, args.img_width, num_class, rnn_hidden=config.rnn_hidden)
    crnn.load(ckp_path)
    crnn.eval()

    # image_predict_with_boxes
    bbox_paths = os.listdir(bbox_dir)
    pbar = tqdm(bbox_paths, desc="Predict")
    for bboxes_filename in pbar:
        # Load image
        img_path = os.path.join(img_dir, bboxes_filename.split(".txt")[0] + ".jpg")
        img = Image.open(img_path).convert('L')

        # Load bboxes
        bboxes_path = os.path.join(bbox_dir, bboxes_filename)
        lines = open(bboxes_path, 'r').readlines()
        bboxes = [line.strip().split(',') for line in lines]

        file = open(bboxes_path, "w", encoding='utf-8')
        for bbox in bboxes:
            bbox = bbox[:8]
            coordinate = [int(i) for i in bbox]
            x, y, w, h = get_xywh(coordinate)
            img_crop = img.crop((x, y, x + w, y + h))

            img_crop = process_img(img_crop, img_width=args.img_width, img_height=args.img_height, img_channel=config.img_channel)
            img_crop = img_crop.unsqueeze(0)

            log_probs = crnn(img_crop)
            preds = ctc_decode(log_probs.numpy(), method=args.decode_method, beam_size=args.beam_size, label2char=LABEL2CHAR)

            prediction = "".join(preds[0])
            bbox.append(prediction)
            line = ",".join(bbox) + "\n"
            file.write(line)

            pbar.set_description("Evaluate - pred_text:{}".format(prediction))
        file.close()
    pbar.close()


def main():
    predict_dataset = PredictDataset(img_paths=args.images,
                                     img_height=args.img_height,
                                     img_width=args.img_width,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=args.cpu_workers)

    num_class = len(LABEL2CHAR) + 1
    crnn = CRNN(config.img_channel, args.img_height, args.img_width, num_class, rnn_hidden=config.rnn_hidden)
    if args.reload_checkpoint[-3:] == ".pt" or args.reload_checkpoint[-4:] == ".pth":
        import torch
        crnn.load_state_dict(torch.load(args.reload_checkpoint, map_location="cpu"))
    else:
        crnn.load(args.reload_checkpoint)

    preds = predict(crnn, predict_dataset, LABEL2CHAR, decode_method=args.decode_method, beam_size=args.beam_size)

    show_result(args.images, preds)


if __name__ == '__main__':
    try:
        jt.flags.use_cuda = not args.cpu
        # pass
    except:
        pass
    print(f'use_cuda: {jt.flags.use_cuda}')

    if args.bbox_dir:
        image_predict_with_boxes()
        # image_predict_with_boxes_using_dataset()
    else:
        main()


# Commands
# python src/predict.py -r ./checkpoints/CRNN.pkl --images ./demo/*.jpg
