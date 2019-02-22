import os
import sys

import rastervision as rv

import tile2vec as t2v

from .data import (get_training_scenes,
                   get_predict_scenes)

class Tile2VecExperiments(rv.ExperimentSet):
    def exp_tile2vec(self,
                     root_uri,
                     data_dir,
                     chip_size=100,
                     model_id='default',
                     predict_output_dir=None,
                     bands=None,
                     test=False):

        if bands is None:
            bands = [0,1,2]
        else:
            bands = list(map(lambda x: int(x), bands.split(',')))

        task = rv.TaskConfig.builder(t2v.TILE2VEC_TASK) \
                            .with_chip_size(chip_size) \
                            .build()

        scenes = get_training_scenes(data_dir)
        val_scenes = scenes

        batch_size = 50
        epochs = 1000
        epoch_size = 1000
        epoch_save_rate = 50
        if test:
            print("Running test, EPOCHS = 1")
            epochs = 3
            epoch_size = 10
            epoch_save_rate = 5
            scenes = scenes[:2]
            val_scenes = scenes[:1]

        backend = rv.BackendConfig.builder(t2v.TILE2VEC_BACKEND) \
                                  .with_task(task) \
                                  .with_scenes(scenes) \
                                  .with_batch_size(batch_size) \
                                  .with_epochs(epochs) \
                                  .with_epoch_size(epoch_size) \
                                  .with_epoch_save_rate(epoch_save_rate) \
                                  .with_pretrained_model('s3://raster-vision-rob-dev/tile2vec/train/iowa-wind-turbines/model.ckpt') \
                                  .build()

        dataset = rv.DatasetConfig.builder() \
                                  .with_validation_scenes(val_scenes) \
                                  .build()

        evaluator = rv.EvaluatorConfig.builder(t2v.TILE2VEC_EVALUATOR).build()

        experiment = rv.ExperimentConfig.builder() \
                                        .with_id('iowa-wind-turbines-2') \
                                        .with_root_uri(root_uri) \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_dataset(dataset) \
                                        .with_evaluator(evaluator) \
                                        .with_predict_key('iowa-wind-turbines-2-epoch350') \
                                        .with_eval_key('iowa-wind-turbines-2-epoch350')

        if predict_output_dir:
            experiment = experiment.with_predict_uri(predict_output_dir)

        return experiment.build()

if __name__ == '__main__':
    rv.main()
