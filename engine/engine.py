import os
import time
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from loguru import logger
from utils.dataset import tokenize
from utils.misc import (AverageMeter, ProgressMeter,)

import utils.misc as utils
import utils.loss_utils as loss_utils
import utils.eval_utils as eval_utils

def train(train_loader, model, optimizer, scheduler, scaler, epoch, args):
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr = AverageMeter('Lr', ':1.6f')
    loss_meter = AverageMeter('Loss', ':2.4f')

    loss_bbox_meter = AverageMeter('Loss_bbox', ':2.4f')
    loss_giou_meter = AverageMeter('Loss_giou', ':2.4f')

    miou_meter = AverageMeter('mIoU', ':2.4f')
    accu_meter = AverageMeter('Accu', ':2.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, lr, loss_meter, loss_bbox_meter, loss_giou_meter, miou_meter, accu_meter],
        prefix="Training: Epoch=[{}/{}] ".format(epoch, args.epochs))

    model.train()
    time.sleep(2)
    end = time.time()

    for i, (image, text, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # data
        image = image.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        # forward
        #with amp.autocast():
        pred = model(image, text, target)
        loss_dict = loss_utils.trans_vg_loss(pred, target)
        losses = sum(loss_dict[k] for k in loss_dict.keys())

        # backward
        optimizer.zero_grad()
        losses.backward()
        if args.max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()

        for name, param in model.named_parameters():
            if param.requires_grad :
                if param.grad is None:
                    print(f"{name} has no gradient.")  # 输出没有梯度的参数名称
        # metric
        miou, accu = eval_utils.trans_vg_eval_val(pred, target)
        dist.all_reduce(losses.detach())
        dist.all_reduce(miou)
        dist.all_reduce(accu)
        losses = losses / dist.get_world_size()
        miou = miou / dist.get_world_size()
        accu = accu / dist.get_world_size()

        loss_meter.update(losses.item(), image.size(0))
        loss_bbox_meter.update(loss_dict["loss_bbox"].item(), image.size(0))
        loss_giou_meter.update(loss_dict["loss_giou"].item(), image.size(0))
        miou_meter.update(torch.mean(miou).item(), image.size(0))
        accu_meter.update(accu.item(), image.size(0))
        lr.update(scheduler.get_last_lr()[-1])
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i + 1)



@torch.no_grad()
def validate(val_loader, model, epoch, args):
    pred_box_list = []
    gt_box_list = []
    model.eval()
    time.sleep(2)
    for imgs, texts, target in val_loader:
        # data
        imgs = imgs.cuda(non_blocking=True)
        texts = texts.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # inference
        preds = model(imgs, texts)
        pred_box_list.append(preds)
        gt_box_list.append(target)
        
    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)
    miou, accu = eval_utils.trans_vg_eval_val(pred_boxes, gt_boxes)

    torch.cuda.synchronize()
    dist.all_reduce(miou)
    dist.all_reduce(accu)
    miou = miou / dist.get_world_size()
    accu = accu / dist.get_world_size()

    head = 'Evaluation: Epoch=[{}/{}] mIoU={:.4f} Accu={:.4f}'.format(
        epoch, args.epochs, torch.mean(miou),  accu)
    logger.info(head)

    return accu


@torch.no_grad()
def inference(test_loader, model, args):
    pred_box_list = []
    gt_box_list = []   
    
    tbar = tqdm(test_loader, desc='Inference:', ncols=100)
    model.eval()
    time.sleep(2)
    for _, batch in enumerate(tbar):

        img_data, text_data, target, param = batch
        batch_size = img_data.size(0)
        # copy to GPU
        img_data = img_data.cuda(non_blocking=True)
        text_data = text_data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(img_data, text_data)
        pred_box_list.append(output)
        gt_box_list.append(target)

    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)
    
    total_num = gt_boxes.shape[0]
    accu_num = eval_utils.trans_vg_eval_test(pred_boxes, gt_boxes)

    result_tensor = torch.tensor([accu_num, total_num])
    

    accuracy = float(result_tensor[0]) / float(result_tensor[1])
    

    logger.info('=> Metric Calculation <=')
    logger.info('Accu: {:.2f}.'.format(100.*accuracy))

    return accuracy
