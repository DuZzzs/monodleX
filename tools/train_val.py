import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import argparse
import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed


parser = argparse.ArgumentParser(description='End-to-End Monocular 3D Object Detection')
parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
# parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')
parser.add_argument('-vis_res', action='store_true', default=False, help='draw box on image and save')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--ip', default='127.0.0.1', type=str)
parser.add_argument('--sync_bn', action='store_true', default=False, help='whether use sync in DDP mode.')


args = parser.parse_args()


def set_logging():
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    set_random_seed(cfg.get('random_seed', 444))
    log_file = 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = cfg.get('log_dir', 'work_dirs/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = create_logger(os.path.join(log_dir, log_file))
    return cfg, logger, log_dir

def main():
    args.nprocs = torch.cuda.device_count()
    cfg, logger, log_dir = set_logging()
    main_worker(args.local_rank, args.nprocs, args, cfg, logger, log_dir)


def main_worker(local_rank, nprocs, args, cfg, logger, log_dir):
    # distributed
    args.local_rank = local_rank
    args.port = str(2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14 + 2)
    init_method = 'tcp://' + args.ip + ':' + args.port

    cudnn.benchmark = True
    dist.init_process_group(backend='nccl', init_method=init_method, world_size=args.nprocs, rank=local_rank)

    # build dataloader
    assert cfg['dataset']['batch_size'] % nprocs == 0, "batch_size {} must be divisible by number of " \
                                                       "gpus: {}".format(cfg['dataset']['batch_size'], nprocs)
    batch_size_per_gpu = int(cfg['dataset']['batch_size'] / nprocs)
    print("batch size: {}".format(cfg['dataset']['batch_size'] / nprocs))
    train_loader, train_sampler, test_loader, test_sampler  = build_dataloader(cfg['dataset'],
                                                                               batch_size_per_gpu=batch_size_per_gpu,
                                                                               rank=local_rank)

    # build model
    model = build_model(cfg['model'])

    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], model)

    # build lr scheduler
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)
    if local_rank == 0:
        logger.info('###################  Training  ##################')
        logger.info('Batch Size: %d'  % (cfg['dataset']['batch_size']))
        logger.info('Learning Rate: %f'  % (cfg['optimizer']['lr']))
    trainer = Trainer(cfg=cfg['trainer'],
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      log_dir=log_dir,
                      local_rank=local_rank,
                      train_sampler=train_sampler,
                      nprocs=nprocs,
                      args=args)
    trainer.train()

    if local_rank == 0:
        cfg['dataset']['root_dir'] = cfg['tester']['root_dir']
        train_loader, train_sampler, test_loader, test_sampler = build_dataloader(cfg['dataset'],
                                                                                  batch_size_per_gpu=1,
                                                                                  rank=-1)
        logger.info('###################  Evaluation  ##################' )
        tester = Tester(cfg=cfg['tester'],
                        model=model,
                        dataloader=test_loader,
                        logger=logger,
                        log_dir=log_dir)
        tester.test()


if __name__ == '__main__':
    main()