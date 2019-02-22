from copy import deepcopy
from typing import (List, Dict, Tuple, Union)

from google.protobuf import struct_pb2
import rastervision as rv
from rastervision.task import ChipClassification
from rastervision.core.class_map import (ClassMap, ClassItem)
from rastervision.task import (TaskConfig, TaskConfigBuilder)
from rastervision.protos.task_pb2 import TaskConfig as TaskConfigMsg
from rastervision.protos.class_item_pb2 import ClassItem as ClassItemMsg

import tile2vec as t2v
from tile2vec.plugin.task.tile2vec_task import Tile2VecTask


class Tile2VecTaskConfig(TaskConfig):
    def __init__(self,
                 predict_batch_size=10,
                 predict_package_uri=None,
                 chip_size=100,
                 debug=True,
                 predict_debug_uri=None):
        super().__init__(t2v.TILE2VEC_TASK, predict_batch_size=predict_batch_size,
                         predict_package_uri=predict_package_uri, debug=debug,
                         predict_debug_uri=predict_debug_uri)
        self.chip_size = chip_size

    def create_task(self, backend):
        return Tile2VecTask(self, backend)

    def to_proto(self):
        struct = struct_pb2.Struct()
        struct['chip_size'] = self.chip_size

        return TaskConfigMsg(
            task_type=self.task_type,
            predict_batch_size=self.predict_batch_size,
            predict_package_uri=self.predict_package_uri,
            custom_config=struct)

    def save_bundle_files(self, bundle_dir):
        return (self, [])

    def load_bundle_files(self, bundle_dir):
        return self


class Tile2VecTaskConfigBuilder(TaskConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                'chip_size': prev.chip_size,
                'predict_batch_size': prev.predict_batch_size,
                'predict_package_uri': prev.predict_package_uri
            }
        super().__init__(Tile2VecTaskConfig, config)

    def from_proto(self, msg):
        b = super().from_proto(msg)
        conf = msg.custom_config
        return b.with_chip_size(int(conf['chip_size']))

    def with_chip_size(self, chip_size):
        """Set the chip_size for this task.

            Args:
                chip_size: Integer value chip size
        """
        b = deepcopy(self)
        b.config['chip_size'] = chip_size
        return b
