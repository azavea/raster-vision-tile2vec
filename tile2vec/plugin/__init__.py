import rastervision as rv

import tile2vec as t2v

from tile2vec.plugin.data.raster_source.remote_geotiff_source_config import (
    RemoteGeoTiffSourceConfigBuilder
)

from tile2vec.plugin.data.raster_source.default import (
    RemoteGeoTiffSourceDefaultProvider)

from tile2vec.plugin.data.label_store.tile_embeddings_store_config import (
    TileEmbeddingsStoreConfigBuilder)

from tile2vec.plugin.task.tile2vec_task_config import (
    Tile2VecTaskConfigBuilder)

from tile2vec.plugin.backend.tile2vec_backend_config import (
    Tile2VecBackendConfigBuilder)

from tile2vec.plugin.evaluation.tile2vec_evaluator_config import (
    Tile2VecEvaluatorConfigBuilder)

def register_plugin(plugin_registry):
    plugin_registry.register_config_builder(rv.RASTER_SOURCE, t2v.REMOTE_GEOTIFF_SOURCE,
                                            RemoteGeoTiffSourceConfigBuilder)

    plugin_registry.register_config_builder(rv.LABEL_STORE, t2v.TILE_EMBEDDINGS_STORE,
                                            TileEmbeddingsStoreConfigBuilder)

    plugin_registry.register_default_vector_source(RemoteGeoTiffSourceDefaultProvider)

    plugin_registry.register_config_builder(rv.BACKEND, t2v.TILE2VEC_BACKEND,
                                            Tile2VecBackendConfigBuilder)

    plugin_registry.register_config_builder(rv.TASK, t2v.TILE2VEC_TASK,
                                            Tile2VecTaskConfigBuilder)

    plugin_registry.register_config_builder(rv.EVALUATOR, t2v.TILE2VEC_EVALUATOR,
                                            Tile2VecEvaluatorConfigBuilder)
