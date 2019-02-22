import os
import uuid
import shutil
from time import time

import numpy as np
import rastervision as rv
from rastervision.utils.files import (download_if_needed, make_dir,start_sync,
                                      sync_to_dir, sync_from_dir)

from tile2vec.datasets import triplet_dataloader
from tile2vec.tilenet import make_tilenet
from tile2vec.plugin.data.label.tile_embeddings import TileEmbeddings


class Tile2VecBackend(rv.backend.Backend):
    def __init__(self, backend_config, task_config):
        import torch
        self.backend_config = backend_config
        self.task_config = task_config
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            print("GPU DETECTED")
        else:
            print("NO GPU DETECTED")

        self.model = None

    def process_scene_data(self, scene, data, tmp_dir):
        pass

    def process_sceneset_results(self, training_results, validation_results,
                                 tmp_dir):
        pass

    def train(self, tmp_dir):
        """Train a model.
        """
        from tile2vec.datasets import triplet_dataloader
        from tile2vec.tilenet import make_tilenet
        from tile2vec.training import train_triplet_epoch
        import torch
        from torch import optim

        # TODO: Config
        batch_size = self.backend_config.batch_size
        epochs = self.backend_config.epochs
        epoch_size = self.backend_config.epoch_size

        img_type = 'naip'
        bands = 4
        augment = True

        shuffle = False
        num_workers = 1

        z_dim = 512

        lr = 1e-3
        betas = (0.5, 0.999)

        margin = 10
        l2 = 0.01
        print_every = 10000
        save_models = False

        sync_interval = 60

        scenes = list(map(lambda s: s.create_scene(self.task_config, tmp_dir), self.backend_config.scenes))

        # Load dataset
        dataloader = triplet_dataloader(img_type, scenes, self.task_config.chip_size, augment=augment,
                                        batch_size=batch_size, epoch_size=epoch_size,
                                        shuffle=shuffle, num_workers=num_workers)
        print('Dataloader set up complete.')

        # Setup TileNet
        in_channels = len(self.backend_config.scenes[0].raster_source.channel_order)
        if in_channels < 1:
            raise Exception("Must set channel order on RasterSource")

        TileNet = make_tilenet(in_channels=in_channels, z_dim=z_dim)
        if self.cuda:
            TileNet.cuda()

        if self.backend_config.pretrained_model_uri:
            model_path = download_if_needed(self.backend_config.pretrained_model_uri, tmp_dir)
            TileNet.load_state_dict(torch.load(model_path))

        TileNet.train()
        print('TileNet set up complete.')

        # Setup Optimizer
        optimizer = optim.Adam(TileNet.parameters(), lr=lr, betas=betas)
        print('Optimizer set up complete.')

        model_dir = os.path.join(tmp_dir, 'training/model_files')

        make_dir(model_dir)

        sync = start_sync(
            model_dir,
            self.backend_config.training_output_uri,
            sync_interval=sync_interval)
        model_path = None
        with sync:
            print('Begin training.................')
            t0 = time()
            for epoch in range(0, epochs):
                print('Epoch {}'.format(epoch))
                (avg_loss, avg_l_n, avg_l_d, avg_l_nd) = train_triplet_epoch(
                    TileNet, self.cuda, dataloader, optimizer, epoch+1, margin=margin, l2=l2,
                    print_every=print_every, t0=t0)

                if epoch % self.backend_config.epoch_save_rate == 0 or epoch + 1 == epochs:
                    print('Saving model for epoch {}'.format(epoch))
                    model_path = os.path.join(model_dir, 'TileNet_epoch{}.ckpt'.format(epoch))
                    torch.save(TileNet.state_dict(), model_path)
                else:
                    print('Skipping model save for epoch {}'.format(epoch))

        if model_path:
            shutil.copy(model_path, os.path.join(model_dir, 'model.ckpt'))
            # Perform final sync
            sync_to_dir(
                model_dir, self.backend_config.training_output_uri, delete=True)

    def load_model(self, tmp_dir):
        """Load the model in preparation for one or more prediction calls."""
        import torch
        from tile2vec.tilenet import make_tilenet

        if self.model is None:
            model_path = download_if_needed(self.backend_config.model_uri, tmp_dir)

            # TODO: config
            in_channels = len(self.backend_config.scenes[0].raster_source.channel_order)
            z_dim = 512

            self.model = make_tilenet(in_channels=in_channels, z_dim=z_dim)
            if self.cuda: self.model.cuda()
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()

    def predict(self, chips, windows, tmp_dir):
        """Return predictions for a chip using model.

        Args:
            chips: [[height, width, channels], ...] numpy array of chips
            windows: List of boxes that are the windows aligned with the chips.

        Return:
            Labels object containing predictions
        """
        import torch
        from torch.autograd import Variable

        # Ensure model is loaded
        self.load_model(tmp_dir)

        embeddings = TileEmbeddings()
        for tile, window in zip(chips, windows):
            # TODO: Configure this

            # Rearrange to PyTorch order
            tile = np.moveaxis(tile, -1, 0)
            tile = np.expand_dims(tile, axis=0)
            # Scale to [0, 1]
            tile = tile / 255

            tile = torch.from_numpy(tile).float()
            tile = Variable(tile)
            if self.cuda: tile = tile.cuda()
            z = self.model.encode(tile)
            if self.cuda: z = z.cpu()
            embeddings.set_window(window, z.data.numpy()[0])

        return embeddings
