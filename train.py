import argparse
import datetime
import os
import math
import shutil
import sys
import time
import warnings
import random
from functools import partial

import cv2
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data as data
from loguru import logger
from torch.optim.lr_scheduler import MultiStepLR
import utils.config as config
# from lion_pytorch import Lion
from utils.dataset import RefDataset
from engine.engine import train, validate
from model import build_swimvg
from utils.misc import (init_random_seed, set_random_seed, setup_logger,
                        worker_init_fn)

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument("--gpu", default="2,3")
    parser.add_argument('--config',
                        default='path to xxx.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')
    
    # Augmentation options
    parser.add_argument('--aug_blur', action='store_true',
                        help="If true, use gaussian blur augmentation")
    parser.add_argument('--aug_crop', action='store_true',
                        help="If true, use random crop augmentation")
    parser.add_argument('--aug_scale', action='store_true',
                        help="If true, use multi-scale augmentation")
    parser.add_argument('--aug_translate', action='store_true',
                        help="If true, use random translate augmentation")
    
    args = parser.parse_args()

    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    return cfg, args


@logger.catch
def main():
    args, data_args = get_parser()
    args.manual_seed = init_random_seed(args.manual_seed)
    set_random_seed(args.manual_seed, deterministic=False)

    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.ngpus_per_node * args.world_size
    # 使用torch run

    #用process启动
    # processes = []
    # for rank in range(args.world_size):
    #     p = mp.Process(target=main_worker, args=(rank, args))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()
    main_worker(args, data_args)
    # mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args, ))


def main_worker(args, data_args):
    args.exp_name = '_'.join([args.exp_name] + [str(name) for name in [args.ladder_dim, args.nhead, args.dim_ffn, args.multi_stage]])
    #expname里加入时间参数
    # dist.barrier()
    args.exp_name = args.exp_name + datetime.datetime.now().strftime(" %Y-%m-%d-%H-%M-%S")
    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    # local rank & global rank
    args.gpu = local_rank
    args.rank = local_rank
    torch.backends.cudnn.enabled = False
    torch.cuda.set_device(args.gpu)
    

    # logger
    setup_logger(args.output_dir,
                 distributed_rank=args.gpu,
                 filename="train.log",
                 mode="a")

    # dist.init_process_group(backend=args.dist_backend,
    #                         init_method=args.dist_url,
    #                         world_size=args.world_size,
    #                         rank=args.rank)#使用torchrun不需要再这样初始化
    dist.barrier() #保证所有进程都已经初始化完成

    # build model
    model, param_list = build_swimvg(args)
    logger.info(args)
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    logger.info(model)
    model = nn.parallel.DistributedDataParallel(model.cuda(),
                                                device_ids=[args.gpu],
                                                find_unused_parameters=False)
                                                # find_unused_parameters=True)

    # build optimizer & lr scheduler
    optimizer = torch.optim.Adam(param_list, lr=args.base_lr, weight_decay=args.weight_decay)
    # optimizer = Lion(param_list, lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_decay)
    #lr_func = lambda epoch: 0.5 * (1. + math.cos(math.pi * epoch / 65))
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)

    scaler = amp.GradScaler()
    # dist.barrier()
    # build dataset
    args.batch_size = int(args.batch_size / args.ngpus_per_node)
    args.batch_size_val = int(args.batch_size_val / args.ngpus_per_node)
    args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
    train_data = RefDataset(data_root=args.data_root,
                            split_root=args.split_root,
                            dataset=args.dataset,
                            split=args.train_split,
                            mode='train',
                            input_size=args.input_size,
                            word_length=args.word_len,
                            args=data_args)
    val_data = RefDataset(data_root=args.data_root,
                          split_root=args.split_root,
                          dataset=args.dataset,
                          split=args.val_split,
                          mode='val',
                          input_size=args.input_size,
                          word_length=args.word_len,
                          args=data_args)

    # build dataloader
    init_fn = partial(worker_init_fn,
                      num_workers=args.workers,
                      rank=args.rank,
                      seed=args.manual_seed)
    train_sampler = data.distributed.DistributedSampler(train_data, shuffle=True)
    val_sampler = data.distributed.DistributedSampler(val_data, shuffle=False)
    train_loader = data.DataLoader(train_data,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.workers,
                                   pin_memory=True,
                                   worker_init_fn=init_fn,
                                   sampler=train_sampler,
                                   drop_last=True)
    val_loader = data.DataLoader(val_data,
                                 batch_size=args.batch_size_val,
                                 shuffle=False,
                                 num_workers=args.workers_val,
                                 pin_memory=True,
                                 sampler=val_sampler,
                                 drop_last=False)

    best_accu = 0.0
    # resume
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_IoU = checkpoint["best_iou"]
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            raise ValueError(
                "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
                .format(args.resume))
    # start training
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1

        # shuffle loader
        train_sampler.set_epoch(epoch_log)

        # train
        train(train_loader, model, optimizer, scheduler, scaler, epoch_log, args)

        # evaluation
        accu = validate(val_loader, model, epoch_log, args)

        # save model
        if dist.get_rank() == 0:
            lastname = os.path.join(args.output_dir, "last_model.pth")
            torch.save(
                {
                    'epoch': epoch_log,
                    'cur_accu': accu,
                    'best_accu': best_accu,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, lastname)
            if accu >= best_accu:
                best_accu = accu
                bestname = os.path.join(args.output_dir, "best_model.pth")
                shutil.copyfile(lastname, bestname)

        # update lr
        scheduler.step(epoch_log)
        torch.cuda.empty_cache()

    time.sleep(2)

    logger.info("* Best accu={} * ".format(best_accu))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('* Training time {} *'.format(total_time_str))


if __name__ == '__main__':
    main()
    sys.exit(0)
