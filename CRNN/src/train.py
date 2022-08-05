from argparse import ArgumentParser
import time
import config
import math
from tqdm import tqdm
import jittor as jt
from jittor import optim

from datasets import TencentData, LABEL2CHAR, PreTrainedDataset, ICDAR2019, ICDAR2019WEAK
from evaluate import evaluate
from model import CRNN
from ctc_decoder import ctc_decode
from utils import save_model, count_similarity

parser = ArgumentParser()
parser.add_argument("-e",
                    "--epochs",
                    default=config.epoch,
                    type=int,
                    help="max epochs to train [default: 100]",
                    metavar="EPOCHS")
parser.add_argument("-b",
                    "--train_batch_size",
                    default=config.train_batch_size,
                    type=int,
                    help="train batch size [default: 32]",
                    metavar="TRAIN BATCH SIZE")
parser.add_argument("-B",
                    "--valid_batch_size",
                    default=config.valid_batch_size,
                    type=int,
                    help="validation batch size [default: 512]",
                    metavar="VALID BATCH SIZE")
parser.add_argument("-l",
                    "--lr",
                    "--learning_rate",
                    default=config.learning_rate,
                    type=float,
                    help="learning rate [default: 5e-5]",
                    metavar="LEARNING RATE")
parser.add_argument("-O",
                    "--optimizer",
                    default="RMSprop",
                    type=str,
                    choices=["RMSprop"],
                    help="the optimizer [default: RMSprop]",
                    metavar="OPTIMIZER")
parser.add_argument("--valid_interval",
                    default=config.valid_interval,
                    type=int,
                    help="number of batches between each 2 evaluation on validation set [default: 2000]",
                    metavar="VALID INTERVAL")
parser.add_argument("--save_interval",
                    default=config.save_interval,
                    type=int,
                    help="number of batches between each 2 savings of model state_dict [default: 2000]",
                    metavar="SAVE INTERVAL")
parser.add_argument("-n",
                    "--cpu_workers",
                    default=16,
                    type=int,
                    help="number of cpu workers used to load data [default: 16]",
                    metavar="CPU WORKERS")
parser.add_argument("-r",
                    "--reload_checkpoint",
                    default=None,
                    type=str,
                    help="the checkpoint to reload [default: None]",
                    metavar="CHECKPOINT")
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
parser.add_argument("--data_dir",
                    default=None,
                    type=str,
                    help="root directory to dataset",
                    metavar="DATA DIR")
parser.add_argument("--checkpoints_dir",
                    default=config.save_dir,
                    type=str,
                    help="directory to save checkpoints [default: ../checkpoints/]",
                    metavar="CHECKPOINTS DIR")
parser.add_argument("--cpu",
                    action="store_true",
                    help="use cpu for all computation, default to enable cuda when possible")
parser.add_argument("--no_shuffle", action="store_true", help="do not shuffle the dataset when training")
parser.add_argument("--valid_max_iter",
                    default=None,
                    type=int,
                    help="max iterations when evaluating the validation set [default: 100]",
                    metavar="VALID MAX ITER")
parser.add_argument("--decode_method",
                    default="greedy",
                    type=str,
                    choices=["greedy", "beam_search", "prefix_beam_search"],
                    help="decode method (greedy, beam_search or prefix_beam_search) [default: greedy]",
                    metavar="DECODE METHOD")
parser.add_argument("--beam_size", default=10, type=int, help="beam size [default: 10]", metavar="BEAM SIZE")
parser.add_argument("-s", "--seed", default=23, type=int, metavar="SEED", help="random number seed [default: 23]")
parser.add_argument("-d",
                    "--dataset",
                    default="TencentData",
                    type=str,
                    choices=["TencentData", "ICDAR2019", "ICDAR2019WEAK", "PreTrainedDataset"],
                    help="Run training process on a specific dataset")
args = parser.parse_args()


def train_batch(crnn, data, optimizer, criterion):
    crnn.train()
    images, targets, target_lengths, _ = [d for d in data]

    log_probs = crnn(images)

    batch_size = images.size(0)
    input_lengths = jt.int64([log_probs.size(0)] * batch_size)

    loss = criterion(log_probs, targets, input_lengths, target_lengths)
    optimizer.step(loss)
    jt.sync_all(True)
    # Count the correct prediction
    preds = ctc_decode(log_probs.numpy(), method="greedy")
    gts = targets.numpy().tolist()
    target_lengths = target_lengths.numpy().tolist()
    correct_num = 0
    for pred, gt, target_length in zip(preds, gts, target_lengths):
        gt = gt[:target_length]
        # print("pred, gt", pred, gt)
        # if pred == gt:
        if count_similarity(pred, gt) >= config.pred_threshold:
            correct_num += 1

    return loss.item(), correct_num


def main():
    try:
        jt.flags.use_cuda = not args.cpu
    except:
        pass
    print(f'use_cuda: {jt.flags.use_cuda}')

    train_dataset = eval(args.dataset)(root_dir=args.data_dir,
                                       mode='train',
                                       img_height=args.img_height,
                                       img_width=args.img_width,
                                       batch_size=args.train_batch_size,
                                       shuffle=True,
                                       num_workers=args.cpu_workers)
    valid_dataset = eval(args.dataset)(root_dir=args.data_dir,
                                       mode='val',
                                       img_height=args.img_height,
                                       img_width=args.img_width,
                                       batch_size=args.valid_batch_size,
                                       shuffle=False,
                                       num_workers=args.cpu_workers)

    num_class = len(LABEL2CHAR) + 1
    crnn = CRNN(config.img_channel, args.img_height, args.img_width, num_class, rnn_hidden=config.rnn_hidden)

    if args.reload_checkpoint:
        if args.reload_checkpoint[-3:] == ".pt":
            import torch
            crnn.load_state_dict(torch.load(args.reload_checkpoint, map_location="cpu"))
        else:
            crnn.load(args.reload_checkpoint)

    if args.optimizer == "RMSprop":
        optimizer = optim.RMSprop(crnn.parameters(), lr=args.lr)
    else:
        raise RuntimeError(f"Unknown optimizer: {args.optimizer}")

    criterion = jt.CTCLoss(reduction='sum')

    try:
        crnn = train_model(crnn, optimizer, criterion, train_dataset, valid_dataset)
    except Exception as e:
        import traceback
        traceback.print_exc()
    save_model(crnn, args.checkpoints_dir, f'CRNN.pkl')


def train_model(crnn, optimizer, criterion, train_dataset, valid_dataset):
    assert args.save_interval % args.valid_interval == 0
    iteration = 0
    for epoch in range(1, args.epochs + 1):
        print(f'\nepoch: {epoch}')
        tot_train_loss = 0.
        tot_train_count = 0
        tot_train_correct = 0
        num = math.ceil(len(train_dataset) / args.train_batch_size)
        pbar = tqdm(enumerate(train_dataset), total=num, desc="Train")
        for batch_id, train_data in pbar:
            iteration += 1
            loss, correct_num = train_batch(crnn, train_data, optimizer, criterion)
            train_size = train_data[0].size(0)

            tot_train_loss += loss
            tot_train_count += train_size
            tot_train_correct += correct_num

            pbar.set_description(
                "Train - epoch:{}, batch:{}, lr:{:.4}, loss:{:.4}, acc:{:.4}".format(epoch, batch_id, optimizer.lr,
                                                                                     tot_train_loss / tot_train_count,
                                                                                     tot_train_correct / tot_train_count))

            if iteration % args.valid_interval == 0:
                print("Evaluation at iteration: %s" % iteration)
                evaluation = evaluate(crnn,
                                      valid_dataset,
                                      criterion,
                                      max_iter=args.valid_max_iter,
                                      decode_method=args.decode_method,
                                      beam_size=args.beam_size,
                                      )
                pbar.clear()
                eval_loss = round(evaluation['loss'], 4)
                acc = round(evaluation['acc'], 4)
                print(f'valid_evaluation: loss={eval_loss}, acc={acc}')

                if iteration % args.save_interval == 0:
                    save_model(crnn, args.checkpoints_dir,
                               f'crnn_epoch-{epoch}_acc-{acc}_lr-{optimizer.lr}_loss-{eval_loss}.pkl')

        pbar.close()
        print('train_loss: ', tot_train_loss / tot_train_count)
    return crnn


if __name__ == '__main__':
    jt.set_global_seed(args.seed)
    start_time = time.time()
    main()
    end_time = time.time()
    print("Used time: %s seconds" % int(end_time - start_time))


# Commands
# python src/train.py --epoch 100 --save_interval 10000 --train_batch_size 4
# python src/train.py -r ./checkpoints/CRNN.pkl
