from argparse import ArgumentParser
import config
import os
import jittor as jt
from tqdm import tqdm

from datasets import TencentData, LABEL2CHAR, CHAR2LABEL, CHARS
from model import CRNN
from ctc_decoder import ctc_decode

from utils import not_real, count_similarity
import pdb

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset",
                        default="TencentData",
                        type=str,
                        choices=["TencentData"],
                        help="name of the dataset",
                        metavar="DATASET")
    parser.add_argument("-p",
                        "--datasets_path",
                        default=config.datasets_path,
                        type=str,
                        help="parent path of all datasets",
                        metavar="DATASETS PATH")
    parser.add_argument("--no_lmdb", action="store_true", help="do not use lmdb, directly load datasets from file")
    parser.add_argument("-r",
                        "--reload_checkpoint",
                        type=str,
                        help="the checkpoint to reload",
                        required=True,
                        metavar="CHECKPOINT")
    parser.add_argument("-l", "--lexicon_based", action="store_true", help="lexicon based method")
    parser.add_argument("-b",
                        "--eval_batch_size",
                        # default=512,
                        default=1,
                        type=int,
                        help="evaluation batch size [default: 512]",
                        metavar="EVAL BATCH SIZE")
    parser.add_argument("-n",
                        "--cpu_workers",
                        default=16,
                        type=int,
                        help="number of cpu workers used to load data [default: 16]",
                        metavar="CPU WORKERS")
    parser.add_argument("--cpu",
                        action="store_true",
                        help="use cpu for all computation, default to enable cuda when possible")
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
    parser.add_argument("-d",
                        "--decode_method",
                        default="greedy",
                        type=str,
                        choices=["greedy", "beam_search", "prefix_beam_search"],
                        help="decode method (greedy, beam_search or prefix_beam_search) [default: greedy]",
                        metavar="DECODE METHOD")
    parser.add_argument("--beam_size", default=10, type=int, help="beam size [default: 10]", metavar="BEAM SIZE")
    parser.add_argument("-g", "--debug", action="store_true", help="enable debug")
    args = parser.parse_args()


def evaluate(crnn,
             dataset,
             criterion,
             max_iter=None,
             decode_method='beam_search',
             beam_size=10,
             lexicon_based=False,
             debug=False):
    crnn.eval()

    tot_count = 0
    tot_loss = 0
    tot_correct = 0

    output_file_path = config.output_file
    output_file = open(output_file_path, "w", encoding='utf-8')

    with jt.no_grad():
        pbar_total = min(max_iter, len(dataset)) if max_iter else len(dataset)
        pbar = tqdm(enumerate(dataset), total=pbar_total, desc="Evaluate")
        for i, data in pbar:
            if max_iter and i >= max_iter:
                break

            images, targets, target_lengths, img_ids = [d for d in data]

            log_probs = crnn(images)

            batch_size = images.size(0)
            input_lengths = jt.int64([log_probs.size(0)] * batch_size)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            if debug and not_real(loss):
                pdb.set_trace()

            preds = ctc_decode(log_probs.numpy(), method=decode_method, beam_size=beam_size)
            gts = targets.numpy().tolist()
            target_lengths = target_lengths.numpy().tolist()

            tot_count += batch_size
            tot_loss += loss.item()
            for pred, gt, target_length, img_id in zip(preds, gts, target_lengths, img_ids):
                gt = gt[:target_length]
                gt_text = ''.join([LABEL2CHAR[c] for c in gt])
                pred_text = ''.join([LABEL2CHAR[c] for c in pred])

                pred_similarity = count_similarity(pred_text, gt_text)
                is_good_pred = pred_similarity >= config.pred_threshold
                if is_good_pred:
                # if pred == gt:
                    tot_correct += 1

                # output the prediction
                output_text0 = "\niter: %s\nimg_id: %s" % (i, img_id)
                output_text1 = "gt: %s, gt_text: %s" % (gt, gt_text)
                output_text2 = "pred: %s, pred_text: %s" % (pred, pred_text)
                output_text3 = "good prediction: %s (%s)\n" % (is_good_pred, pred_similarity)
                output_file.write("\n".join([output_text0, output_text1, output_text2, output_text3]))

            pbar.set_description("Evaluate - iter:{}, acc:{:.4}".format(i, tot_correct / tot_count))
        pbar.close()

    evaluation = {'loss': tot_loss / tot_count,
                  'acc': tot_correct / tot_count,
                  'correct_count': tot_correct,
                  'total_count': tot_count}
    output_file.write("\nEvaluation Result:\n" + str(evaluation))
    output_file.close()
    return evaluation


def main():
    try:
        jt.flags.use_cuda = not args.cpu
    except:
        pass
    print(f'use_cuda: {jt.flags.use_cuda}')

    dataset_path = os.path.join(args.datasets_path, args.dataset)
    test_dataset = eval(args.dataset)(
        root_dir=dataset_path,
        mode='val',
        img_height=args.img_height,
        img_width=args.img_width,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.cpu_workers)

    num_class = len(LABEL2CHAR) + 1
    crnn = CRNN(config.img_channel, args.img_height, args.img_width, num_class, rnn_hidden=config.rnn_hidden)
    if args.reload_checkpoint[-3:] == ".pt":
        import torch
        crnn.load_state_dict(torch.load(args.reload_checkpoint, map_location="cpu"))
    else:
        crnn.load(args.reload_checkpoint)

    criterion = jt.CTCLoss(reduction='sum')

    evaluation = evaluate(crnn,
                          test_dataset,
                          criterion,
                          decode_method=args.decode_method,
                          beam_size=args.beam_size,
                          lexicon_based=args.lexicon_based,
                          debug=args.debug)

    print('test_evaluation: loss={loss}, acc={acc}'.format(**evaluation))


if __name__ == '__main__':
    main()


# Commands
# python src/evaluate.py -r ./checkpoints/CRNN.pkl
