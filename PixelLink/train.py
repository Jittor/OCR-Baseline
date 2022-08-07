import time
import os
import argparse

import jittor as jt
from jittor import optim
from dataset.datasets import TencentDataLoader
from dataset.transform import build_transform
from models.vgg import VGGPixel
from tools.loss import PixelLinkLoss
from config_pl import cfg
from tqdm import tqdm
from tools import postprocess
from tools.vis import val_draw_labels
from tools.utils import output_prediction, resize_prediction
from jittor.optim import LambdaLR


jt.flags.use_cuda = 1


def adjust_learning_rate(optimizer, epoch):
    if epoch in cfg.schedule:
        optimizer.lr = optimizer.lr*0.1


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    
def val(model, dataloader, val_ckp=None, epoch=0, output=False):
    model.eval()
    if val_ckp:
        model.load(val_ckp)
    true_pos, false_pos, false_neg = [0] * 3
    nb = len(dataloader)
    pbar = tqdm(enumerate(dataloader), total=nb)
    for iter, sample in pbar:
        images, _, _, gts, _, _, img_paths = sample
        
        out_1, out_2 = model(images)

        predict = postprocess.cal_label_on_batch(out_1, out_2)

#         res = postprocess.comp_gt_and_output(predict, gts, 0.5)
        res = postprocess.comp_gt_and_output(predict, gts, 0.3)

        true_pos += res[0]
        false_pos += res[1]
        false_neg += res[2]

        if output:
            prediction = predict[0]
            prediction = resize_prediction(prediction, img_paths[0])
            filename = img_paths[0].split("/")[-1].split(".")[0] + ".txt"
            output_prediction(prediction, filename)

        if not iter % cfg.vis_freq:
            val_draw_labels(images, gts, epoch, iter, predict, img_paths[0].split("/")[-1].split(".")[0])
#             break

    if (true_pos + false_pos) > 0:
        precision = true_pos / (true_pos + false_pos)
    else:
        precision = 0
    if (true_pos + false_neg) > 0:
        recall = true_pos / (true_pos + false_neg)
    else:
        recall = 0
    print("epoch: {} -> TP: {}, FP: {}, FN: {}, precision: {}, recall: {}".format(epoch,
          true_pos, false_pos, false_neg, precision, recall))


def train(model, dataloader, val_dataloader, epoch, optimizer, scheduler, start_epoch=0, exp_name='exp'):
    
    exp_dir = os.path.join(cfg.save_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    if start_epoch > 0:
        model.load(os.path.join(exp_dir, 'epoch_'+str(start_epoch)+'pkl'))

    for i in range(start_epoch, epoch):
        model.train()
        
#         adjust_learning_rate(optimizer, i)
        nb = len(dataloader)
        pbar = tqdm(enumerate(dataloader), total=nb//cfg.batch_size)
        for iter, sample in pbar:
            images, pixel_masks, neg_pixel_masks, gts, pixel_pos_weights, link_masks, _ = sample
            # val_draw_labels(images, gts, i, iter)
            out_1, out_2 = model(images)
            loss_instance = PixelLinkLoss()
            
            pixel_loss_pos, pixel_loss_neg = loss_instance.pixel_loss(
                out_1, pixel_masks, neg_pixel_masks, pixel_pos_weights)
            pixel_loss = pixel_loss_pos + pixel_loss_neg
            link_loss_pos, link_loss_neg = loss_instance.link_loss(
                out_2, link_masks)
            link_loss = link_loss_pos + link_loss_neg
            losses = cfg.pixel_weight * pixel_loss + cfg.link_weight * link_loss

            optimizer.step(losses)
        
            pbar.set_description("epoch:{}, iter:{}, lr:{:.4}, pixel_loss:{:.4}, link_loss:{:.4}, total_loss:{:.4}".format(
                i, iter, get_lr(optimizer), pixel_loss, link_loss, losses))

        if not i % cfg.save_freq and i > 0:
            model.save(os.path.join(exp_dir, "epoch_" + str(i) + ".pkl"))

        if not i % cfg.val_freq and i > 0:
            val(model, val_dataloader, epoch=i)
            
        scheduler.step()

    model.save(os.path.join(exp_dir, "PixelLink.pkl"))


def main():
    transform = build_transform(cfg, mode=args.mode)
    val_transform = build_transform(cfg, mode="val")
    
    train_dataloader = TencentDataLoader(cfg, mode="train", transform=transform, batch_size=cfg.batch_size,
                                         shuffle=True, num_workers=cfg.num_workers)
    val_dataloader = TencentDataLoader(cfg, mode="val", transform=val_transform, batch_size=1,
                                       shuffle=False, num_workers=cfg.num_workers)

    model = VGGPixel()
    # model = VGGPixel(pretrained=True)
    
    optimizer = optim.SGD(model.parameters(),
                          lr=cfg.lr,
                          momentum=cfg.momentum,
                          weight_decay=cfg.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.99 ** epoch+1e-4)

    if args.mode == 'train':
        train(model, train_dataloader, val_dataloader, cfg.epoch, optimizer, scheduler,
              args.resume_epoch, args.exp_name)
    elif args.mode == 'val':
        val(model, val_dataloader, args.val_ckp, output=args.output)
    else:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mode', type=str, default='train',
                        help='train val or test')
    parser.add_argument('--resume_epoch', type=int, default=0,
                        help='True for resume, False for train')
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--val_ckp', type=str, default='')
    parser.add_argument('--output', action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    main()
    end_time = time.time()
    print("Used time: %s seconds" % int(end_time-start_time))


# Commands
# python train.py
# python train.py --mode val --val_ckp "./work_dirs/exp/PixelLink.pkl"
