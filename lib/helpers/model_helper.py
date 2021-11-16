from lib.models.centernet3d import CenterNet3D


def build_model(cfg):
    if cfg['model']['type'] == 'centernet3d':
        return CenterNet3D(backbone=cfg['model']['backbone'], neck=cfg['model']['neck'],
                           num_class=cfg['model']['num_class'], cfg=cfg)
    else:
        raise NotImplementedError("%s model is not supported" % cfg['type'])


