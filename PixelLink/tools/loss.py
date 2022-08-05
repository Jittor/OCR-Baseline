import jittor as jt
from jittor import nn
from config_pl import cfg


class PixelLinkLoss(object):
    def __init__(self):
        pass

    def pixel_loss(self, input, target, neg_pixel_masks, pos_weight):
        batch_size = input.size(0)
        softmax_input = nn.softmax(input, dim=-3)

        self.pixel_cross_entropy = nn.cross_entropy_loss(input, target, reduction='none')
        self.area = jt.int32(target).view(batch_size, -1).sum(dim=1)
        int_area = self.area.int()
        self.area = self.area.float()
        self.pos_pixel_weight = pos_weight
        self.neg_pixel_weight = jt.zeros_like(self.pos_pixel_weight, dtype=jt.uint8)
        self.neg_area = jt.zeros_like(self.area, dtype=jt.int)
        
        for i in range(batch_size):
            wrong_input = softmax_input[i, 0][neg_pixel_masks[i] == 1].view(-1)
            r_pos_area = int_area[i] * cfg.neg_pos_ratio
            if r_pos_area == 0:
                r_pos_area = 10000
            self.neg_area[i] = min(r_pos_area, wrong_input.size(0))

            topk, _ = jt.topk(-wrong_input, int(self.neg_area[i].numpy()))

            self.neg_pixel_weight[i, (softmax_input[i, 0] <= -topk[-1])] = 1
            self.neg_pixel_weight[i] = self.neg_pixel_weight[i] & (neg_pixel_masks[i] == 1)

        self.pixel_weight = self.pos_pixel_weight + self.neg_pixel_weight.float()
        weighted_pixel_cross_entropy_pos = self.pos_pixel_weight * self.pixel_cross_entropy
        weighted_pixel_cross_entropy_pos = weighted_pixel_cross_entropy_pos.view(
            batch_size, -1)

        weighted_pixel_cross_entropy_neg = self.neg_pixel_weight.float() * \
            self.pixel_cross_entropy
        weighted_pixel_cross_entropy_neg = weighted_pixel_cross_entropy_neg.view(
            batch_size, -1)
        weighted_pixel_cross_entropy = weighted_pixel_cross_entropy_neg + \
            weighted_pixel_cross_entropy_pos

        return [jt.mean(jt.sum(weighted_pixel_cross_entropy_pos, dim=1) /
                (self.area + self.neg_area.float())),
                jt.mean(jt.sum(weighted_pixel_cross_entropy_neg, dim=1) /
                (self.area + self.neg_area.float())),
                ]

    def link_loss(self, input, target, neighbors=8):
        batch_size = input.size(0)
        self.pos_link_weight = (
            target == 1).float() * self.pos_pixel_weight.unsqueeze(1).expand(-1, neighbors, -1, -1)
        self.neg_link_weight = (
            target == 0).float() * self.pos_pixel_weight.unsqueeze(1).expand(-1, neighbors, -1, -1)
        sum_pos_link_weight = jt.sum(
            self.pos_link_weight.view(batch_size, -1), dim=1)
        sum_neg_link_weight = jt.sum(
            self.neg_link_weight.view(batch_size, -1), dim=1)

        self.link_cross_entropy = jt.empty(self.pos_link_weight.size(), dtype=self.pos_link_weight.dtype)

        for i in range(neighbors):
            assert input.size(1) == 16
            this_input = input[:, [2 * i, 2 * i + 1]]
            this_target = target[:, i]
            if list(this_target.shape)[1] == 1:
                this_target = this_target.squeeze(1)
            self.link_cross_entropy[:, i] = nn.cross_entropy_loss(
                this_input, this_target, reduction='none')

        loss_link_pos = jt.empty(self.pos_link_weight.size(),dtype=self.pos_link_weight.dtype)
        loss_link_neg = jt.empty(self.neg_link_weight.size(),dtype=self.neg_link_weight.dtype)
    
        for i in range(batch_size):
            if sum_pos_link_weight[i] == 0:
                loss_link_pos[i] = 0
            else:
                loss_link_pos[i] = self.pos_link_weight[i] * \
                    self.link_cross_entropy[i] / sum_pos_link_weight[i]
                
            if sum_neg_link_weight[i] == 0:
                loss_link_neg[i] = 0
            else:
                loss_link_neg[i] = self.neg_link_weight[i] * \
                    self.link_cross_entropy[i] / sum_neg_link_weight[i]

        loss_link_pos = jt.sum(loss_link_pos.view(batch_size, -1), dim=1)
        loss_link_neg = jt.sum(loss_link_neg.view(batch_size, -1), dim=1)
        return jt.mean(loss_link_pos), jt.mean(loss_link_neg)
