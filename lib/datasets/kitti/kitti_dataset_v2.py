import os
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
import cv2

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))

from lib.datasets.utils import angle2class
from lib.datasets.utils import gaussian_radius
from lib.datasets.utils import draw_umich_gaussian
from lib.datasets.kitti.kitti_utils import get_objects_from_label
from lib.datasets.kitti.kitti_utils import Calibration
from lib.datasets.kitti.kitti_utils import get_affine_transform
from lib.datasets.kitti.kitti_utils import affine_transform, approx_proj_center, draw_umich_gaussian_2D
from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result
from lib.datasets.kitti.kitti_eval_python.eval import get_distance_eval_result
import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti
from lib.datasets.utils import draw_projected_box3d


class KITTI_Dataset_v2(data.Dataset):
    def __init__(self, split, cfg, label_dir=None):
        # basic configuration
        self.root_dir = cfg.get('root_dir', '../../data/KITTI')
        self.split = split
        self.num_classes = 3
        self.max_objs = 50
        self.class_name = ['Pedestrian', 'Car', 'Cyclist']
        self.cls2id = {'Pedestrian': 0, 'Car': 1, 'Cyclist': 2}
        self.resolution = np.array([1280, 384])  # W * H
        self.use_3d_center = cfg.get('use_3d_center', True)
        self.writelist = cfg.get('writelist', ['Car'])
        # anno: use src annotations as GT, proj: use projected 2d bboxes as GT
        self.bbox2d_type = cfg.get('bbox2d_type', 'anno')
        assert self.bbox2d_type in ['anno', 'proj']
        self.meanshape = cfg.get('meanshape', False)
        self.class_merging = cfg.get('class_merging', False)
        self.use_dontcare = cfg.get('use_dontcare', False)

        if self.class_merging:
            self.writelist.extend(['Van', 'Truck'])
        if self.use_dontcare:
            self.writelist.extend(['DontCare'])


        # data split loading
        assert self.split in ['train', 'val', 'trainval', 'test']
        flag = 'training' if self.split == 'train' else 'training'
        self.split_file = os.path.join(self.root_dir, flag, 'ImageSets', self.split + '.txt')
        self.idx_list = [x.strip() for x in open(self.split_file).readlines()]
        # print(self.idx_list)

        # path configuration
        self.data_dir = os.path.join(self.root_dir, 'testing' if split == 'test' else 'training')
        # self.data_dir = os.path.join(self.root_dir, 'object', 'testing' if split == 'test' else 'training')
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.depth_dir = os.path.join(self.data_dir, 'depth')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        if label_dir:
            self.label_dir = os.path.join(label_dir)  # for test
        else:
            self.label_dir = os.path.join(self.data_dir, 'label_2')  # for train

        # data augmentation configuration
        self.data_augmentation = True if split in ['train', 'trainval'] else False
        self.random_flip = cfg.get('random_flip', 0.5)
        self.random_crop = cfg.get('random_crop', 0.5)
        self.scale = cfg.get('scale', 0.4)
        self.shift = cfg.get('shift', 0.1)

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.cls_mean_size = np.array([[1.76255119, 0.66068622, 0.84422524],
                                       [1.52563191, 1.62856739, 3.52588311],
                                       [1.73698127, 0.59706367, 1.76282397]], dtype=np.float32)  # H*W*L
        if not self.meanshape:
            self.cls_mean_size = np.zeros_like(self.cls_mean_size, dtype=np.float32)

        # others
        self.downsample = 4
        self.features_size = self.resolution // self.downsample

        # x_I
        self.consider_outside_objs = True
        self.enable_edge_fusion = True
        self.max_edge_length = ((self.features_size[0] + self.features_size[1]) * 2).item()
        self.filter_annos = True
        self.filter_params = [0.7, 10]
        self.adjust_edge_heatmap = True
        self.edge_heatmap_ratio = 0.5

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return Image.open(img_file)


    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)


    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return Calibration(calib_file)

    def eval(self, results_dir, logger):
        logger.info("==> Loading detections and GTs...")
        img_ids = [int(id) for id in self.idx_list]
        dt_annos = kitti.get_label_annos(results_dir)
        gt_annos = kitti.get_label_annos(self.label_dir, img_ids)

        test_id = {'Car': 0, 'Pedestrian':1, 'Cyclist': 2}

        logger.info('==> Evaluating (official) ...')
        for category in self.writelist:
            results_str, results_dict = get_official_eval_result(gt_annos, dt_annos, test_id[category])
            logger.info(results_str)

    def get_edge_utils(self, image_size, pad_size, down_ratio=4):
        img_w, img_h = image_size  # original image size

        # output feature map boundary
        x_min, y_min = np.ceil(pad_size[0] / down_ratio), np.ceil(pad_size[1] / down_ratio)
        x_max, y_max = (pad_size[0] + img_w - 1) // down_ratio, (pad_size[1] + img_h - 1) // down_ratio

        step = 1
        # boundary idxs
        edge_indices = []

        # left
        y = torch.arange(y_min, y_max, step)  # [y_min, y_max)  y_max will show in bottom
        x = torch.ones(len(y)) * x_min

        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = torch.unique(edge_indices_edge, dim=0)
        edge_indices.append(edge_indices_edge)

        # bottom
        x = torch.arange(x_min, x_max, step)
        y = torch.ones(len(x)) * y_max

        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = torch.unique(edge_indices_edge, dim=0)
        edge_indices.append(edge_indices_edge)

        # right
        y = torch.arange(y_max, y_min, -step)
        x = torch.ones(len(y)) * x_max

        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = torch.unique(edge_indices_edge, dim=0).flip(dims=[0])
        edge_indices.append(edge_indices_edge)

        # top
        x = torch.arange(x_max, x_min - 1, -step)  # right to left
        y = torch.ones(len(x)) * y_min

        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = torch.unique(edge_indices_edge, dim=0).flip(dims=[0])  # 这里flip没起作用？
        edge_indices.append(edge_indices_edge)

        # concatenate
        edge_indices = torch.cat([index.long() for index in edge_indices], dim=0)

        return edge_indices  # (n,2)

    def __len__(self):
        return self.idx_list.__len__()


    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        # image loading
        img = self.get_image(index)
        img_size = np.array(img.size)
        features_size = self.features_size    # W * H

        # data augmentation for image
        center = np.array(img_size) / 2
        aug_scale, crop_size = 1.0, img_size
        random_crop_flag, random_flip_flag = False, False
        if self.data_augmentation:
            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # if np.random.random() < self.random_crop:
            #     random_crop_flag = True
            #     aug_scale = np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
            #     crop_size = img_size * aug_scale
            #     center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
            #     center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)

        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        img = img.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)

        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W

        if self.enable_edge_fusion:
            # generate edge_indices for the edge fusion module
            pad_size = np.array([0, 0])
            input_edge_indices = np.zeros([self.max_edge_length, 2], dtype=np.int64)
            edge_indices = self.get_edge_utils(self.resolution, pad_size, down_ratio=self.downsample).numpy()
            input_edge_count = edge_indices.shape[0]
            input_edge_indices[:edge_indices.shape[0]] = edge_indices
            input_edge_count = input_edge_count - 1  # (0,0) count twice

        info = {'img_id': index,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size/features_size}
        if self.enable_edge_fusion:
            info["edge_len"] = input_edge_count
            info["edge_indices"] = input_edge_indices

        if self.split == 'test' or self.split == 'val':
            return img, img, info   # img / placeholder(fake label) / info


        #  ============================   get labels   ==============================
        objects = self.get_label(index)
        calib = self.get_calib(index)

        # computed 3d projected box
        if self.bbox2d_type == 'proj':
            for object in objects:
                object.box2d_proj = np.array(calib.corners3d_to_img_boxes(object.generate_corners3d()[None, :])[0][0], dtype=np.float32)
                object.box2d = object.box2d_proj.copy()

        # data augmentation for labels
        if random_flip_flag:
            for object in objects:
                [x1, _, x2, _] = object.box2d
                object.box2d[0],  object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                object.alpha = np.pi - object.alpha
                object.ry = np.pi - object.ry
                if object.alpha > np.pi:  object.alpha -= 2 * np.pi  # check range
                if object.alpha < -np.pi: object.alpha += 2 * np.pi
                if object.ry > np.pi:  object.ry -= 2 * np.pi
                if object.ry < -np.pi: object.ry += 2 * np.pi

        # labels encoding
        heatmap = np.zeros((self.num_classes, features_size[1], features_size[0]), dtype=np.float32) # C * H * W
        size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
        offset_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
        depth = np.zeros((self.max_objs, 1), dtype=np.float32)
        heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
        heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
        src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        dimension = np.ones((self.max_objs, 3), dtype=np.float32)
        offset_3d = np.zeros((self.max_objs, 2), dtype=np.float32)
        indices = np.zeros((self.max_objs), dtype=np.int64)
        mask_2d = np.zeros((self.max_objs), dtype=np.int64)
        mask_3d = np.zeros((self.max_objs), dtype=np.int64)
        trunc_mask = np.zeros((self.max_objs), dtype=np.int64)

        object_num = len(objects) if len(objects) < self.max_objs else self.max_objs
        for i in range(object_num):
            # filter objects by writelist
            if objects[i].cls_type not in self.writelist:
                continue

            # filter inappropriate samples
            # if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
            if objects[i].pos[-1] <= 1:
                continue

            # ignore the samples beyond the threshold [hard encoding]
            threshold = 65
            if objects[i].pos[-1] > threshold:
                continue

            # process 2d bbox & get 2d center
            bbox_2d = objects[i].box2d.copy()

            # add affine transformation for 2d boxes.
            bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
            bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)

            # filter some unreasonable annotations
            float_truncation = objects[i].trucation
            if self.filter_annos:
                if float_truncation >= self.filter_params[0] \
                    or (bbox_2d[2:] - bbox_2d[:2]).min() <= self.filter_params[1]:
                    continue
            input_bbox_2d = bbox_2d.copy()

            # modify the 2d bbox according to pre-compute downsample ratio
            bbox_2d[:] /= self.downsample

            # process 3d bbox & get 3d center
            center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)  # W * H
            center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
            center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
            center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
            center_3d = center_3d[0]  # shape adjustment
            if random_flip_flag:  # random flip for center3d
                center_3d[0] = img_size[0] - center_3d[0]
            center_3d = affine_transform(center_3d.reshape(-1), trans)

            proj_inside_img = (0 <= center_3d[0] <= self.resolution[0] - 1) & (0 <= center_3d[1] <= self.resolution[1] - 1)
            approx_center = False
            if not proj_inside_img:
                if self.consider_outside_objs:
                    approx_center = True
                    input_center_2d = (input_bbox_2d[:2] + input_bbox_2d[2:]) / 2
                    center_3d, edge_index = approx_proj_center(center_3d, input_center_2d.reshape(1, 2),
                                                                        self.resolution)
                else:
                    continue

            center_3d /= self.downsample

            # generate the center of gaussian heatmap [optional: 3d center or 2d center]
            center_heatmap = center_3d.astype(np.int32) if self.use_3d_center else center_2d.astype(np.int32)
            if center_heatmap[0] < 0 or center_heatmap[0] >= features_size[0]: continue
            if center_heatmap[1] < 0 or center_heatmap[1] >= features_size[1]: continue

            # generate heatmap
            cls_id = self.cls2id[objects[i].cls_type]
            if self.adjust_edge_heatmap and approx_center:
                # for outside objects, generate 1-dimensional heatmap
                bbox_width = min(center_heatmap[0] - bbox_2d[0], bbox_2d[2] - center_heatmap[0])
                bbox_height = min(center_heatmap[1] - bbox_2d[1], bbox_2d[3] - center_heatmap[1])
                radius_x, radius_y = bbox_width * self.edge_heatmap_ratio, bbox_height * self.edge_heatmap_ratio
                radius_x, radius_y = max(0, int(radius_x)), max(0, int(radius_y))
                assert min(radius_x, radius_y) == 0
                heatmap[cls_id] = draw_umich_gaussian_2D(heatmap[cls_id], center_heatmap, radius_x, radius_y)
                w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
            else:
                # generate the radius of gaussian heatmap
                w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
                radius = gaussian_radius((w, h))
                radius = max(0, int(radius))

                if objects[i].cls_type in ['Van', 'Truck', 'DontCare']:
                    draw_umich_gaussian(heatmap[1], center_heatmap, radius)
                    continue

                draw_umich_gaussian(heatmap[cls_id], center_heatmap, radius)

            # encoding 2d/3d offset & 2d size
            indices[i] = center_heatmap[1] * features_size[0] + center_heatmap[0]
            offset_2d[i] = center_2d - center_heatmap  # x_b - x_c or x_b - x_I
            size_2d[i] = 1. * w, 1. * h

            # encoding depth
            depth[i] = objects[i].pos[-1] * aug_scale

            # encoding heading angle
            heading_angle = objects[i].alpha
            heading_bin[i], heading_res[i] = angle2class(heading_angle)

            # encoding 3d offset & size_3d
            offset_3d[i] = center_3d - center_heatmap
            src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
            mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
            size_3d[i] = src_size_3d[i] - mean_size
            dimension[i] = src_size_3d[i] - mean_size

            mask_2d[i] = 1
            mask_3d[i] = 0 if random_crop_flag else 1
            trunc_mask[i] = int(approx_center)

        # collect return data
        inputs = img
        targets = {'depth': depth,
                   'size_2d': size_2d,
                   'heatmap': heatmap,
                   'offset_2d': offset_2d,
                   'indices': indices,
                   'size_3d': size_3d,
                   'dimension': dimension,
                   'src_size_3d': src_size_3d,
                   'offset_3d': offset_3d,
                   'heading_bin': heading_bin,
                   'heading_res': heading_res,
                   'mask_2d': mask_2d,
                   'mask_3d': mask_3d,
                   'trunc_mask': trunc_mask}
        if self.enable_edge_fusion:
            targets['edge_len'] = input_edge_count
            targets['edge_indices'] = input_edge_indices
        info = {'img_id': index,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size/features_size}
        return inputs, targets, info




if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import yaml

    # cfg = {'root_dir': '../../../data/KITTI',
    #        'random_flip':0.0, 'random_crop':1.0, 'scale':0.8, 'shift':0.1, 'use_dontcare': False,
    #        'class_merging': False, 'writelist':['Pedestrian', 'Car', 'Cyclist'], 'use_3d_center':False}
    save_dir = "../../../experiments/example/work_dirs/input_check"
    config_file = "../../../experiments/example/test_kitti_v2.yaml"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    cfg = yaml.load(open(config_file, 'r'), Loader=yaml.Loader)
    cfg = cfg['dataset']
    dataset = KITTI_Dataset_v2('train', cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    print(dataset.writelist)

    for batch_idx, (inputs, targets, info) in enumerate(dataloader):
        print("batch_idx: {}, idx_list: {}".format(batch_idx, dataset.idx_list[batch_idx]))
        # test image
        img = inputs[0].numpy().transpose(1, 2, 0)
        img = (img * dataset.std + dataset.mean) * 255
        img = Image.fromarray(img.astype(np.uint8))
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # print(targets['size_3d'][0][0])

        # test heatmap
        # heatmap = targets['heatmap'][0]  # image id
        # heatmap = Image.fromarray(heatmap[0].numpy() * 255)  # cats id
        # heatmap.show()

        # test heatmap
        heatmap = targets['heatmap'][0]  # image id
        hm = heatmap[1].numpy() * 255  # cats id
        hm = cv2.applyColorMap(hm.astype(np.uint8), cv2.COLORMAP_JET)
        hm = cv2.resize(hm, None, None, fx=dataset.downsample, fy=dataset.downsample, interpolation=cv2.INTER_LINEAR)
        out_img = cv2.addWeighted(img, 0.6, hm, 0.4, gamma=0)
        # heatmap = cv2.cvtColor(np.asarray(heatmap), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, dataset.idx_list[batch_idx] + "_hm.jpg"), out_img)
        if batch_idx > 100:
            break
        # break

    # print ground truth fisrt
    objects = dataset.get_label(0)
    for object in objects:
        print(object.to_kitti_format())
