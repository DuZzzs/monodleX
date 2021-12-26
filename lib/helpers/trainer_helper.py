import os
import tqdm

import torch
import numpy as np
import torch.nn as nn

from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint
from lib.losses.centernet_loss import compute_centernet3d_loss
import torch.distributed as dist
# from torch.utils.tensorboard import SummaryWriter


def reduce_mean(tensor, nprocs):
    rt =tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger,
                 log_dir,
                 local_rank,
                 train_sampler,
                 nprocs,
                 args):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cuda', local_rank)
        self.log_dir = log_dir
        self.local_rank = local_rank
        self.train_sampler = train_sampler
        self.nprocs = nprocs

        torch.cuda.set_device(local_rank)
        model.cuda(local_rank)

        # loading pretrain/resume model
        if cfg.get('pretrain_model'):
            assert os.path.exists(cfg['pretrain_model'])
            load_checkpoint(model=model,
                            optimizer=None,
                            filename=cfg['pretrain_model'],
                            map_location=self.device,
                            logger=self.logger)

        if cfg.get('resume_model', None):
            assert os.path.exists(cfg['resume_model'])
            # self.epoch = load_checkpoint(model=self.model.to(self.device),
            self.epoch = load_checkpoint(model=model,
                                         optimizer=self.optimizer,
                                         filename=cfg['resume_model'],
                                         map_location=self.device,
                                         logger=self.logger)
            self.lr_scheduler.last_epoch = self.epoch - 1

        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
            self.logger.info("Using SyncBatchNorm.")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                          find_unused_parameters=True)
        self.model = model
        if self.local_rank == 0:
            self.writer = SummaryWriter(log_dir + "runs/")


    def train(self):
        start_epoch = self.epoch
        if self.local_rank == 0:
            progress_bar = tqdm.tqdm(range(start_epoch, self.cfg['max_epoch']), dynamic_ncols=True, leave=True, desc='epochs')
        for epoch in range(start_epoch, self.cfg['max_epoch']):
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            self.train_sampler.set_epoch(epoch)

            # train one epoch
            self.train_one_epoch(epoch)
            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
                # self.warmup_lr_scheduler.step(epoch)
            else:
                self.lr_scheduler.step()
                # self.lr_scheduler.step(epoch)

            # save trained model
            if self.local_rank == 0 and (self.epoch % self.cfg['save_frequency']) == 0:
                os.makedirs(os.path.join(self.log_dir, 'checkpoints'), exist_ok=True)
                ckpt_name = os.path.join(self.log_dir, 'checkpoints', 'checkpoint_epoch_%d' % self.epoch)
                save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch), ckpt_name)
            if self.local_rank == 0:
                progress_bar.update()

        return None


    def train_one_epoch(self, epoch):
        self.model.train()
        loss_items = {}
        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=(self.epoch+1 == self.cfg['max_epoch']), desc='iters')
        for batch_idx, (inputs, targets, _) in enumerate(self.train_loader):
            # inputs = inputs.to(self.device)
            inputs = inputs.cuda(self.local_rank, non_blocking=True)
            for key in targets.keys():
                # targets[key] = targets[key].to(self.device)
                targets[key] = targets[key].cuda(self.local_rank, non_blocking=True)

            # train one batch
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            total_loss, stats_batch, dist_dict = compute_centernet3d_loss(outputs, targets)

            torch.distributed.barrier()
            for key_stat, value_stat in dist_dict.items():
                dist_dict[key_stat] = reduce_mean(value_stat, self.nprocs).item()

            total_loss.backward()
            self.optimizer.step()
            if self.local_rank == 0:
                # self.update_loss_item(loss_items, stats_batch)
                self.update_loss_item(loss_items, dist_dict)

                progress_bar.update()
        if self.local_rank == 0:
            progress_bar.close()
            self.print_loss_items(loss_items, epoch)

            num_batch = len(self.train_loader)
            # self.writer("Loss/seg_loss", loss_items['seg'] / num_batch, epoch)
            # self.writer("Loss/offset2d_loss", loss_items['offset2d'] / num_batch, epoch)
            # self.writer("Loss/size2d_loss", loss_items['size2d'] / num_batch, epoch)
            # self.writer("Loss/offset3d_loss", loss_items['offset3d'] / num_batch, epoch)
            # self.writer("Loss/depth_loss", loss_items['depth'] / num_batch, epoch)
            # self.writer("Loss/size3d_loss", loss_items['size3d'] / num_batch, epoch)
            # self.writer("Loss/heading_loss", loss_items['heading'] / num_batch, epoch)

    def update_loss_item(self, loss_items, stats_batch):
        # cache loss item of this batch
        if not loss_items:
            for idx, key in enumerate(stats_batch.keys()):
                loss_items[key] = 0.0
        for idx, key in enumerate(stats_batch.keys()):
            loss_items[key] += stats_batch[key]

    def print_loss_items(self, loss_items, epoch):
        self.logger.info("epoch : {} \n".format(epoch + 1))
        num_batch = len(self.train_loader)
        for key, value in loss_items.items():
            self.logger.info("{} loss: {}, ".format(key, value / num_batch))
        self.logger.info("\n")




