import torch
import numpy as np
from torch.utils.data import DataLoader
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset

# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataloader(cfg, workers=4, batch_size_per_gpu=1, rank=-1):
    # perpare dataset
    if cfg['type'] == 'KITTI':
        train_set = KITTI_Dataset(split='train', cfg=cfg)
        test_set = KITTI_Dataset(split='val', cfg=cfg)
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])
    # TODO: Both single gpu and multi gpu modes are supported
    if rank != -1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None
    # prepare dataloader
    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size_per_gpu,
                              num_workers=workers,
                              worker_init_fn=my_worker_init_fn,
                              # shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              sampler=train_sampler)

    if rank != -1:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
    else:
        test_sampler = None
    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size_per_gpu,
                             num_workers=workers,
                             worker_init_fn=my_worker_init_fn,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False,
                             sampler=test_sampler)

    return train_loader, train_sampler, test_loader, test_sampler
