import os
from copy import deepcopy

from google.protobuf import struct_pb2
import rastervision as rv
from rastervision.evaluation import (EvaluatorConfig, EvaluatorConfigBuilder)
from rastervision.protos.evaluator_pb2 import EvaluatorConfig as EvaluatorConfigMsg

import tile2vec as t2v
from tile2vec.plugin.evaluation.tile2vec_evaluator import Tile2VecEvaluator

class Tile2VecEvaluatorConfig(EvaluatorConfig):
    def __init__(self, index_output_uri=None, window_list_uri=None, nearest_output_uri=None, compute_nearest=False):
        super().__init__(t2v.TILE2VEC_EVALUATOR)
        self.index_output_uri = index_output_uri
        self.window_list_uri = window_list_uri
        self.nearest_output_uri = nearest_output_uri
        self.compute_nearest = compute_nearest

    def to_proto(self):
        struct = struct_pb2.Struct()
        struct['index_output_uri'] = self.index_output_uri
        struct['window_list_uri'] = self.window_list_uri
        struct['nearest_output_uri'] = self.nearest_output_uri
        struct['compute_nearest'] = self.compute_nearest

        return EvaluatorConfigMsg(
            evaluator_type=self.evaluator_type, custom_config=struct)

    def create_evaluator(self):
        return Tile2VecEvaluator(self)

    def update_for_command(self, command_type, experiment_config,
                           context=None):
        if command_type == rv.EVAL:
            if not self.index_output_uri:
                self.index_output_uri = os.path.join(experiment_config.eval_uri,
                                               'embeddings.idx')

            if not self.window_list_uri:
                self.window_list_uri = os.path.join(experiment_config.eval_uri,
                                               'scene-windows.csv')
            if not self.nearest_output_uri:
                self.nearest_output_uri = os.path.join(experiment_config.eval_uri,
                                               'nearest.csv')
    def report_io(self, command_type, io_def):
        if command_type == rv.EVAL:
            io_def.add_output(self.index_output_uri)
            io_def.add_output(self.window_list_uri)
            io_def.add_output(self.nearest_output_uri)


class Tile2VecEvaluatorConfigBuilder(EvaluatorConfigBuilder):
    def __init__(self, prev=None):
        self.config = {}
        if prev:
            self.config = {
                'index_output_uri': prev.index_output_uri,
                'window_list_uri': prev.index_output_uri,
                'nearest_output_uri': prev.nearest_output_uri,
                'compute_nearest': prev.compute_nearest
            }
        super().__init__(Tile2VecEvaluatorConfig, self.config)

    def from_proto(self, msg):
        conf = msg.custom_config
        b = self.with_index_output_uri(conf['index_output_uri'])
        b = b.with_window_list_uri(conf['window_list_uri'])
        b = b.with_nearest_output_uri(conf['nearest_output_uri'])
        b = b.with_compute_nearest(conf['compute_nearest'])
        return b

    def with_index_output_uri(self, index_output_uri):
        b = deepcopy(self)
        b.config['index_output_uri'] = index_output_uri
        return b

    def with_window_list_uri(self, window_list_uri):
        b = deepcopy(self)
        b.config['window_list_uri'] = window_list_uri
        return b

    def with_nearest_output_uri(self, nearest_output_uri):
        b = deepcopy(self)
        b.config['nearest_output_uri'] = nearest_output_uri
        return b

    def with_compute_nearest(self, compute_nearest):
        b = deepcopy(self)
        b.config['compute_nearest'] = compute_nearest
        return b
