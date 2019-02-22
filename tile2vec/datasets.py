from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import glob
import os
import sys
import numpy as np
import random
from rastervision.core import Box
from tile2vec.data_utils import clip_and_scale_image

def find_neighbor_window(extent, window, chip_size):
    """
     _ _ _
    |0|1|2|
    |3| |4|
    |5|6|7|
    """
    neighbors = [0,1,2,3,4,5,6,7]
    random.shuffle(neighbors)
    for neighbor_id in neighbors:
        if neighbor_id == 0:
            n = Box(ymin=window.ymin - chip_size - 1,
                    xmin=window.xmin - chip_size - 1,
                    ymax=window.ymin - 1,
                    xmax=window.xmin - 1)
            if n.intersection(extent) == n:
                if not (n.get_height() == 100 and n.get_width() == 100):
                    raise Exception("{},{}".format(n.get_height(), n.get_width()))
                return n
        if neighbor_id == 1:
            n = Box(ymin=window.ymin - chip_size - 1,
                    xmin=window.xmin,
                    ymax=window.ymin - 1,
                    xmax=window.xmax)
            if n.intersection(extent) == n:
                if not (n.get_height() == 100 and n.get_width() == 100):
                    raise Exception("{},{}".format(n.get_height(), n.get_width()))
                return n
        if neighbor_id == 2:
            n = Box(ymin=window.ymin - chip_size - 1,
                    xmin=window.xmax + 1,
                    ymax=window.ymin - 1,
                    xmax=window.xmax + chip_size + 1)
            if n.intersection(extent) == n:
                if not (n.get_height() == 100 and n.get_width() == 100):
                    raise Exception("{},{}".format(n.get_height(), n.get_width()))
                return n
        if neighbor_id == 3:
            n = Box(ymin=window.ymin,
                    xmin=window.xmin - chip_size - 1,
                    ymax=window.ymax,
                    xmax=window.xmin - 1)
            if n.intersection(extent) == n:
                if not (n.get_height() == 100 and n.get_width() == 100):
                    raise Exception("{},{}".format(n.get_height(), n.get_width()))
                return n
        if neighbor_id == 4:
            n = Box(ymin=window.ymin,
                    xmin=window.xmax + 1,
                    ymax=window.ymax,
                    xmax=window.xmax + chip_size + 1)
            if n.intersection(extent) == n:
                if not (n.get_height() == 100 and n.get_width() == 100):
                    raise Exception("{},{}".format(n.get_height(), n.get_width()))
                return n
        if neighbor_id == 5:
            n = Box(ymin=window.ymax + 1,
                    xmin=window.xmin - chip_size - 1,
                    ymax=window.ymax + chip_size + 1,
                    xmax=window.xmin - 1)
            if n.intersection(extent) == n:
                if not (n.get_height() == 100 and n.get_width() == 100):
                    raise Exception("{},{}".format(n.get_height(), n.get_width()))
                return n
        if neighbor_id == 6:
            n = Box(ymin=window.ymax + 1,
                    xmin=window.xmin,
                    ymax=window.ymax + chip_size + 1,
                    xmax=window.xmax)
            if n.intersection(extent) == n:
                if not (n.get_height() == 100 and n.get_width() == 100):
                    raise Exception("{},{}".format(n.get_height(), n.get_width()))
                return n
        if neighbor_id == 7:
            n = Box(ymin=window.ymax + 1,
                    xmin=window.xmax + 1,
                    ymax=window.ymax + chip_size + 1,
                    xmax=window.xmax + chip_size + 1)
            if n.intersection(extent) == n:
                if not (n.get_height() == 100 and n.get_width() == 100):
                    raise Exception("{},{}".format(n.get_height(), n.get_width()))
                return n

    raise Exception('No neighbor: Extent {}, window {}'.format(extent, neighbor))


class TileTripletsDataset(Dataset):

    def __init__(self, scenes, chip_size, epoch_size, transform=None):
        self.scenes = scenes # Raster Vision Scenes
        self.chip_size = chip_size
        self.epoch_size = epoch_size
        self.transform = transform

    def get_random_chip(self, scene, include_neighbor=False):
        """Gets a random chip from a scene."""
        extent = scene.raster_source.get_extent()
        window = extent.make_random_square(self.chip_size)
        a = scene.raster_source.get_chip(window)
        if not include_neighbor:
            return a
        else:
            n_window = find_neighbor_window(extent, window, self.chip_size)
            n = scene.raster_source.get_chip(n_window)
            return (a, n)

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        """ Randomly gets a (a, n, d) triplet.

        Assumes that a random chip window in a randomly selected scene
        that is not equal to the anchor scene will produce a 'distant'
        chip.
        """

        print('.', end='', flush=True)

        scenes = random.sample(self.scenes, k=2)
        with scenes[0].activate():
            a, n = self.get_random_chip(scenes[0], include_neighbor=True)

        with scenes[1].activate():
            d = self.get_random_chip(scenes[1])

        a = np.moveaxis(a, -1, 0)
        n = np.moveaxis(n, -1, 0)
        d = np.moveaxis(d, -1, 0)
        sample = {'anchor': a, 'neighbor': n, 'distant': d}
        if self.transform:
            sample = self.transform(sample)
        return sample


### TRANSFORMS ###

# Not needed, RasterSource includes channel_order
# class GetBands(object):
#     """
#     Gets the first X bands of the tile triplet.
#     """
#     def __init__(self, bands):
#         assert bands >= 0, 'Must get at least 1 band'
#         self.bands = bands

#     def __call__(self, sample):
#         a, n, d = (sample['anchor'], sample['neighbor'], sample['distant'])
#         # Tiles are already in [c, w, h] order
#         a, n, d = (a[:self.bands,:,:], n[:self.bands,:,:], d[:self.bands,:,:])
#         sample = {'anchor': a, 'neighbor': n, 'distant': d}
#         return sample

class RandomFlipAndRotate(object):
    """
    Does data augmentation during training by randomly flipping (horizontal
    and vertical) and randomly rotating (0, 90, 180, 270 degrees). Keep in mind
    that pytorch samples are CxWxH.
    """
    def __call__(self, sample):
        a, n, d = (sample['anchor'], sample['neighbor'], sample['distant'])
        # Randomly horizontal flip
        if np.random.rand() < 0.5: a = np.flip(a, axis=2).copy()
        if np.random.rand() < 0.5: n = np.flip(n, axis=2).copy()
        if np.random.rand() < 0.5: d = np.flip(d, axis=2).copy()
        # Randomly vertical flip
        if np.random.rand() < 0.5: a = np.flip(a, axis=1).copy()
        if np.random.rand() < 0.5: n = np.flip(n, axis=1).copy()
        if np.random.rand() < 0.5: d = np.flip(d, axis=1).copy()
        # Randomly rotate
        rotations = np.random.choice([0, 1, 2, 3])
        if rotations > 0: a = np.rot90(a, k=rotations, axes=(1,2)).copy()
        rotations = np.random.choice([0, 1, 2, 3])
        if rotations > 0: n = np.rot90(n, k=rotations, axes=(1,2)).copy()
        rotations = np.random.choice([0, 1, 2, 3])
        if rotations > 0: d = np.rot90(d, k=rotations, axes=(1,2)).copy()
        sample = {'anchor': a, 'neighbor': n, 'distant': d}
        return sample

class ClipAndScale(object):
    """
    Clips and scales bands to between [0, 1] for NAIP, RGB, and Landsat
    satellite images. Clipping applies for Landsat only.
    """
    def __init__(self, img_type):
        assert img_type in ['naip', 'rgb', 'landsat']
        self.img_type = img_type

    def __call__(self, sample):
        a, n, d = (clip_and_scale_image(sample['anchor'], self.img_type),
                   clip_and_scale_image(sample['neighbor'], self.img_type),
                   clip_and_scale_image(sample['distant'], self.img_type))
        sample = {'anchor': a, 'neighbor': n, 'distant': d}
        return sample

class ToFloatTensor(object):
    """
    Converts numpy arrays to float Variables in Pytorch.
    """
    def __call__(self, sample):
        a, n, d = (torch.from_numpy(sample['anchor']).float(),
            torch.from_numpy(sample['neighbor']).float(),
            torch.from_numpy(sample['distant']).float())
        sample = {'anchor': a, 'neighbor': n, 'distant': d}
        return sample

### TRANSFORMS ###


def triplet_dataloader(img_type, scenes, chip_size, augment=True,
                       batch_size=4, epoch_size=10, shuffle=False, num_workers=4):
    """
    Returns a DataLoader with either NAIP (RGB/IR), RGB, or Landsat tiles.
    Turn shuffle to False for producing embeddings that correspond to original
    tiles.
    """
    assert img_type in ['landsat', 'rgb', 'naip']
    transform_list = []
    # if img_type in ['landsat', 'naip']: transform_list.append(GetBands(bands))
    transform_list.append(ClipAndScale(img_type))
    if augment: transform_list.append(RandomFlipAndRotate())
    transform_list.append(ToFloatTensor())
    transform = transforms.Compose(transform_list)
    dataset = TileTripletsDataset(scenes, chip_size=chip_size, epoch_size=epoch_size,
                                  transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers)
    return dataloader
