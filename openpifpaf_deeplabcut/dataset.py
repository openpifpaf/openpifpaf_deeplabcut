from collections import defaultdict
import copy
import logging
import os

import h5py
import numpy as np
import torch.utils.data
from PIL import Image

import openpifpaf


LOG = logging.getLogger(__name__)


class DeepLabCutDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, ann_file, *, preprocess=None):
        super().__init__()

        self.image_dir = image_dir
        annotation_file = h5py.File(ann_file, 'r')
        LOG.debug('%s', list(annotation_file.keys()))
        LOG.debug('%s', list(annotation_file['df_with_missing'].keys()))
        self.annotations = list(annotation_file['df_with_missing']['table'])
        LOG.debug('%s', self.annotations)

        LOG.info('Images: %d', len(self.annotations))
        self.preprocess = preprocess or openpifpaf.transforms.EVAL_TRANSFORM

    def __getitem__(self, index):
        entry = copy.deepcopy(self.annotations[index])
        LOG.debug('dataset entry = %s', entry)
        file_name = os.path.basename(str(entry[0]).strip('\''))

        keypoints = np.array(entry[1], np.float32).reshape(-1, 2)
        keypoints = np.concatenate((
            keypoints,
            np.full((keypoints.shape[0], 1), 2.0, dtype=np.float32),
        ), axis=1)
        # set confidence to zero when coordinate is nan:
        keypoints[np.isnan(keypoints[:, 0]), 2] = 0.0

        anns = [{
            'category_id': 1,
            'keypoints': keypoints,
        }]

        local_file_path = os.path.join(self.image_dir, file_name)
        LOG.debug('local file path: %s', local_file_path)
        with open(local_file_path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        meta = {
            'dataset_index': index,
            'file_name': file_name,
            'local_file_path': local_file_path,
        }

        # preprocess image and annotations
        image, anns, meta = self.preprocess(image, anns, meta)

        LOG.debug(meta)
        return image, anns, meta

    def __len__(self):
        return len(self.annotations)
