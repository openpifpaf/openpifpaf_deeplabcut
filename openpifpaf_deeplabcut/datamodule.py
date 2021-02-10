import argparse
import logging

import torch
import yaml

import openpifpaf

from .dataset import DeepLabCutDataset

LOG = logging.getLogger(__name__)


class DeepLabCut(openpifpaf.datasets.DataModule):
    config = 'data-dlc-reaching/config.yaml'
    train_annotations = 'data-dlc-reaching/CollectedData_Mackenzie.h5'
    val_annotations = 'data-dlc-reaching/CollectedData_Mackenzie.h5'
    train_image_dir = 'data-dlc-reaching/'
    val_image_dir = 'data-dlc-reaching/'

    square_edge = 385
    augmentation = True
    upsample_stride = 1
    bmin = 0.1

    def __init__(self):
        super().__init__()

        with open(self.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        LOG.debug('config: %s', config)
        keypoints = config['bodyparts']
        skeleton = [(keypoints.index(j1) + 1, keypoints.index(j2) + 1)
                    for j1, j2 in config['skeleton']]
        sigmas = [0.1 for _ in keypoints]

        cif = openpifpaf.headmeta.Cif('cif', 'cocokp',
                                      keypoints=keypoints,
                                      sigmas=sigmas,
                                      draw_skeleton=skeleton)
        caf = openpifpaf.headmeta.Caf('caf', 'cocokp',
                                      keypoints=keypoints,
                                      sigmas=sigmas,
                                      skeleton=skeleton)

        cif.upsample_stride = self.upsample_stride
        caf.upsample_stride = self.upsample_stride
        self.head_metas = [cif, caf]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module DeepLabCut')

        group.add_argument('--deeplabcut-train-annotations',
                           default=cls.train_annotations)
        group.add_argument('--deeplabcut-val-annotations',
                           default=cls.val_annotations)
        group.add_argument('--deeplabcut-train-image-dir',
                           default=cls.train_image_dir)
        group.add_argument('--deeplabcut-val-image-dir',
                           default=cls.val_image_dir)

        group.add_argument('--deeplabcut-square-edge',
                           default=cls.square_edge, type=int,
                           help='square edge of input images')
        assert cls.augmentation
        group.add_argument('--deeplabcut-no-augmentation',
                           dest='deeplabcut_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--deeplabcut-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')
        group.add_argument('--deeplabcut-bmin',
                           default=cls.bmin, type=float,
                           help='bmin')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # deeplabcut specific
        cls.train_annotations = args.deeplabcut_train_annotations
        cls.val_annotations = args.deeplabcut_val_annotations
        cls.train_image_dir = args.deeplabcut_train_image_dir
        cls.val_image_dir = args.deeplabcut_val_image_dir

        cls.square_edge = args.deeplabcut_square_edge
        cls.augmentation = args.deeplabcut_augmentation
        cls.upsample_stride = args.deeplabcut_upsample
        cls.bmin = args.deeplabcut_bmin

    def _preprocess(self):
        encoders = [openpifpaf.encoder.Cif(self.head_metas[0], bmin=self.bmin),
                    openpifpaf.encoder.Caf(self.head_metas[1], bmin=self.bmin)]
        if len(self.head_metas) > 2:
            encoders.append(openpifpaf.encoder.Caf(self.head_metas[2], bmin=self.bmin))

        if not self.augmentation:
            return openpifpaf.transforms.Compose([
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.RescaleAbsolute(self.square_edge),
                openpifpaf.transforms.CenterPad(self.square_edge),
                openpifpaf.transforms.EVAL_TRANSFORM,
                openpifpaf.transforms.Encoders(encoders),
            ])

        rescale_t = openpifpaf.transforms.RescaleRelative(
            scale_range=(0.4, 2.0), power_law=True, stretch_range=(0.75, 1.33))

        return openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            rescale_t,
            openpifpaf.transforms.Crop(self.square_edge, use_area_of_interest=True),
            openpifpaf.transforms.CenterPad(self.square_edge),
            openpifpaf.transforms.TRAIN_TRANSFORM,
            openpifpaf.transforms.Encoders(encoders),
        ])

    def train_loader(self):
        train_data = DeepLabCutDataset(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self._preprocess(),
        )
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug and self.augmentation,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def val_loader(self):
        val_data = DeepLabCutDataset(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._preprocess(),
        )
        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=not self.debug and self.augmentation,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)
