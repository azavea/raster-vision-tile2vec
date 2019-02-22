from os.path import join
from copy import deepcopy
from google.protobuf import struct_pb2
from google.protobuf import json_format

import rastervision as rv
from rastervision.backend import (BackendConfig, BackendConfigBuilder)
from rastervision.protos.backend_pb2 import BackendConfig as BackendConfigMsg
from rastervision.protos.scene_pb2 import SceneConfig as SceneConfigMsg
from rastervision.filesystem.filesystem import ProtobufParseException

import tile2vec as t2v
from tile2vec.plugin.backend.tile2vec_backend import Tile2VecBackend

# TODO: Merge into RV, check if dict.
def load_json_config(json_dict, message, fs=None):
    """Load a JSON-formatted protobuf config file.

    Args:
        json_dict: JSON dict of message
        message: (google.protobuf.message.Message) empty protobuf message of
            to load the config into. The type needs to match the content of
            uri.
        fs: Optional FileSystem to use.

    Returns:
        the same message passed as input with fields filled in from uri

    Raises:
        ProtobufParseException if uri cannot be parsed
    """
    try:
        return json_format.ParseDict(json_dict, message)
    except json_format.ParseError as e:
        error_msg = ('Problem parsing protobuf from {}. '.format(json_dict) +
                     'You might need to run scripts/compile')
        raise ProtobufParseException(error_msg) from e


class Tile2VecBackendConfig(BackendConfig):
    def __init__(self,
                 scenes,
                 batch_size=50,
                 epochs=50,
                 epoch_size=50,
                 epoch_save_rate=5,
                 training_output_uri=None,
                 pretrained_model_uri=None,
                 model_uri=None):

        super().__init__(t2v.TILE2VEC_BACKEND)

        self.scenes = scenes
        self.batch_size = batch_size
        self.epochs = epochs
        self.epoch_size = epoch_size
        self.epoch_save_rate = epoch_save_rate
        self.training_output_uri = training_output_uri
        self.pretrained_model_uri = pretrained_model_uri
        self.model_uri = model_uri

    def to_proto(self):
        struct = struct_pb2.Struct()
        struct['scenes'] = list(map(lambda x: json_format.MessageToDict(x.to_proto()), self.scenes))
        struct['batch_size'] = self.batch_size
        struct['epochs'] = self.epochs
        struct['epoch_size'] = self.epoch_size
        struct['epoch_save_rate'] = self.epoch_save_rate
        struct['training_output_uri'] = self.training_output_uri
        struct['model_uri'] = self.model_uri

        msg = BackendConfigMsg(
            backend_type=self.backend_type, custom_config=struct)

        if self.pretrained_model_uri:
            msg.MergeFrom(
                BackendConfigMsg(
                    pretrained_model_uri=self.pretrained_model_uri))
        return msg

    def create_backend(self, task_config):
        return Tile2VecBackend(self, task_config)

    def update_for_command(self, command_type, experiment_config,
                           context=None):
        super().update_for_command(command_type, experiment_config,
                                            context)
        if command_type == rv.TRAIN:
            if not self.training_output_uri:
                self.training_output_uri = experiment_config.train_uri
            if not self.model_uri:
                self.model_uri = join(self.training_output_uri,
                                      'model.ckpt')

    def report_io(self, command_type, io_def):
        super().report_io(command_type, io_def)
        if command_type == rv.TRAIN:
            io_def.add_output(self.model_uri)

        if command_type in [rv.PREDICT, rv.BUNDLE]:
            if not self.model_uri:
                io_def.add_missing('Missing model_uri.')
            else:
                io_def.add_input(self.model_uri)

    def save_bundle_files(self, bundle_dir):
        if not self.model_uri:
            raise rv.ConfigError('model_uri is not set.')
        local_path, base_name = self.bundle_file(self.model_uri, bundle_dir)
        new_config = self.to_builder() \
                         .with_model_uri(base_name) \
                         .build()
        return (new_config, [local_path])

    def load_bundle_files(self, bundle_dir):
        if not self.model_uri:
            raise rv.ConfigError('model_uri is not set.')
        local_model_uri = join(bundle_dir, self.model_uri)
        return self.to_builder() \
                   .with_model_uri(local_model_uri) \
                   .build()


class Tile2VecBackendConfigBuilder(BackendConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                'scenes': prev.scenes,
                'batch_size': prev.batch_size,
                'epochs': prev.epochs,
                'epoch_size': prev.epoch_size,
                'epoch_save_rate': prev.epoch_save_rate,
                'model_uri': prev.model_uri,
                'training_output_uri': prev.training_output_uri
            }
        self.require_task = prev is None
        super().__init__(t2v.TILE2VEC_BACKEND, Tile2VecBackendConfig, config)

    def from_proto(self, msg):
        b = super().from_proto(msg)
        conf = msg.custom_config

        scenes = []
        for scene_struct in conf['scenes']:
            scene_json = json_format.MessageToDict(scene_struct)
            scenes.append(rv.SceneConfig.from_proto(load_json_config(scene_json, SceneConfigMsg())))
        b = b.with_scenes(scenes)
        b = b.with_batch_size(int(conf['batch_size']))
        b = b.with_epochs(int(conf['epochs']))
        b = b.with_epoch_size(int(conf['epoch_size']))
        b = b.with_epoch_save_rate(int(conf['epoch_save_rate']))
        b = b.with_model_uri(conf['model_uri'])
        b = b.with_training_output_uri(conf['training_output_uri'])

        # b = b.with_img_type(conf['img_type'])
        # b = b.with_band_count(conf['band_count'])

        # img_type = 'naip'
        # bands = 4
        # augment = True
        # batch_size = 50
        # shuffle = True
        # num_workers = 4

        # z_dim = 512

        # lr = 1e-3
        # betas = (0.5, 0.999)

        # epochs = 50
        # margin = 10
        # l2 = 0.01
        # print_every = 10000
        # save_models = False

        return b

    def _applicable_tasks(self):
        return [t2v.TILE2VEC_TASK]

    def _process_task(self):
        return self

    def with_scenes(self, scenes):
        """Defines the scenes that should be sampled for training.
        Also used for determining band_count via the first scene.
        """
        b = deepcopy(self)
        b.config['scenes'] = scenes
        return b

    def with_batch_size(self, batch_size):
        """Defines the name of the model file that will be created for
        this model after training.

        """
        b = deepcopy(self)
        b.config['batch_size'] = batch_size
        return b

    def with_epochs(self, epochs):
        b = deepcopy(self)
        b.config['epochs'] = epochs
        return b

    def with_epoch_size(self, epoch_size):
        b = deepcopy(self)
        b.config['epoch_size'] = epoch_size
        return b

    def with_epoch_save_rate(self, epoch_save_rate):
        b = deepcopy(self)
        b.config['epoch_save_rate'] = epoch_save_rate
        return b

    def with_model_uri(self, model_uri):
        """Defines the name of the model file that will be created for
        this model after training.

        """
        b = deepcopy(self)
        b.config['model_uri'] = model_uri
        return b

    def with_training_output_uri(self, training_output_uri):
        """Whither goes the training output?

            Args:
                training_output_uri: The location where the training
                    output will be stored.

        """
        b = deepcopy(self)
        b.config['training_output_uri'] = training_output_uri
        return b
