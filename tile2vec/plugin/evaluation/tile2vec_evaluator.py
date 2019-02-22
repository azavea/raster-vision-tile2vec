import os

import rastervision as rv
from rastervision.evaluation import Evaluator
import numpy as np
import pandas as pd
from rastervision.utils.files import (upload_or_copy, download_if_needed)

class Tile2VecEvaluator(Evaluator):
    def __init__(self, config):
        self.config = config

    def process(self, scenes, tmp_dir):
        import faiss
        import torch
        z_dim = 512
        index = faiss.IndexFlatL2(z_dim)
        if torch.cuda.is_available():
            print('USING GPU')
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        else:
            print('NOT USING GPU')
        scenes_and_windows = []
        for scene in scenes:
            print('Embedding scene {}'.format(scene.id))
            embeddings = []
            for window, embedding in scene.prediction_label_store.get_labels().get_embeddings():
                scenes_and_windows.append((scene, window))
                embeddings.append(embedding)
            print('.' * len(embeddings), end='', flush=True)
            print(np.array(embeddings).shape)
            index.add(np.array(embeddings))
        print('.')

        # Write index
        print('Writing index...')
        index_path = os.path.join(tmp_dir, 'embeddings.idx')
        if torch.cuda.is_available():
            faiss.write_index(faiss.index_gpu_to_cpu(index), index_path)
        else:
            faiss.write_index(index, index_path)
        if not os.path.exists(index_path):
            raise Exception('Did not write index to {}'.format(index_path))
        else:
            print('Wrote index to {}'.format(index_path))
            print('Uploading to {}'.format(self.config.index_output_uri))
        upload_or_copy(index_path, self.config.index_output_uri)

        # Write scene list.
        def make_scene_list_row(src_scene_id, src_window, i):
            return { 'idx': i,
                     'src_scene_id': src_scene_id,
                     'src_window_xmin': src_window.xmin,
                     'src_window_ymin': src_window.ymin,
                     'src_window_xmax': src_window.xmax,
                     'src_window_ymax': src_window.ymax }

        print('Writing scene list...')
        rows = []
        for (i, (scene, window)) in enumerate(scenes_and_windows):
            rows.append(make_scene_list_row(scene.raster_source.uris[0], window, i))
        df = pd.DataFrame(rows)

        path = os.path.join(tmp_dir, 'scene-windows.csv')
        df.to_csv(path)

        upload_or_copy(path, self.config.window_list_uri)

        if self.config.compute_nearest:
            # Calculate nearest.
            k = 4
            results = []
            for scene in scenes:
                print('Finding {} nearest for scene {}'.format(k, scene.id))
                windows = []
                embeddings = []
                for window, embedding in scene.prediction_label_store.get_labels().get_embeddings():
                    windows.append(window)
                    embeddings.append(embedding)
                print('.' * len(embeddings), end='', flush=True)
                D, I = index.search(np.array(embeddings), k)
                for i, nearest_idx in enumerate(I):
                    nearest = []
                    for idx in nearest_idx:
                        near_scene, near_window = scenes_and_windows[idx]
                        nearest.append((near_scene.id, near_window))
                    results.append((scene.id, windows[i], nearest))
            print('.')

            # Write results
            def make_nearest_row(src_scene_id, src_window, near_scene_id, near_window, near_rank):
                return {'src_scene_id': src_scene_id,
                        'src_window_xmin': src_window.xmin,
                        'src_window_ymin': src_window.ymin,
                        'src_window_xmax': src_window.xmax,
                        'src_window_ymax': src_window.ymax,
                        'near_scene_id': near_scene_id,
                        'near_window_xmin': near_window.xmin,
                        'near_window_ymin': near_window.ymin,
                        'near_window_xmax': near_window.xmax,
                        'near_window_ymax': near_window.ymax,
                        'near_rank': near_rank
                }


            print('Writing results...')
            rows = []
            for (src_scene_id, src_window, neighbors) in results:
                for near_rank, (near_scene_id, near_window) in enumerate(neighbors):
                    rows.append(make_nearest_row(src_scene_id, src_window, near_scene_id,
                                     near_window, near_rank))
            df = pd.DataFrame(rows)

            path = os.path.join(tmp_dir, 'nearest.csv')
            df.to_csv(path)

            upload_or_copy(path, self.config.nearest_output_uri)
