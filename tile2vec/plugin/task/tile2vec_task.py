from abc import abstractmethod

import numpy as np
import logging
import rastervision as rv
from rastervision.core.training_data import TrainingData
from rastervision.core.box import Box

from tile2vec.plugin.data.label.tile_triplets import TileTriplets

log = logging.getLogger(__name__)


def get_random_sample_train_windows(chip_size, extent,
                                    samples_per_scene, filter_window):
    windows = []
    i = 0
    while i < samples_per_scene:
        candidate_window = extent.make_random_square(chip_size)
        if not filter_window(candidate_window):
            continue
        windows.append(candidate_window)
        i += 1
    return windows


class Tile2VecTask(rv.task.Task):
    """Functionality for a specific machine learning task.

    This should be subclassed to add a new task, such as object detection
    """

    def __init__(self, task_config, backend):
        """Construct a new Task.

        Args:
            task_config: TaskConfig
            backend: Backend
        """
        self.config = task_config
        self.backend = backend

    def get_train_windows(self, scene):
        """Training windows are created at train time, so this is a noop"""
        # def filter_window(window):
        #     if scene.aoi_polygons:
        #         windows = Box.filter_by_aoi(windows, scene.aoi_polygons)
        #     return windows

        # extent = scene.raster_source.get_extent()

        # return get_random_sample_train_windows(
        #     self.config.chip_size, extent,
        #     self.config.samples_per_scene, filter_window)
        return []

    def get_train_labels(self, window, scene):
        # extent = scene.raster_source.get_extent()
        # labels = TileTriplets()
        # near = get_near_window(extent, window)
        # distant = get_distant_window(extent, window)
        # labels.set_anchor(window, near, distant)
        # return labels
        pass

    def make_chips(self, train_scenes, validation_scenes, augmentors, tmp_dir):
        """Override to make training chips command a no-op"""
        log.info("Tile2Vec does not create training chips, skipping...")

    def post_process_predictions(self, labels, scene):
        return labels

    def get_predict_windows(self, extent):
        chip_size = self.config.chip_size
        stride = chip_size
        return extent.get_windows(chip_size, stride)

    def save_debug_predict_image(self, scene, debug_dir_uri):
        pass
