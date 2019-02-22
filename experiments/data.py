import os

import rastervision as rv

import tile2vec as t2v

def get_training_scenes(data_dir):
    fs = rv.filesystem.FileSystem.get_file_system(data_dir)
    tiffs = fs.list_paths(data_dir, 'tif')
    # tiffs = ['s3://raster-vision-iowa-wind-turbines/images/m_4309546_nw_15_1_20170912.tif',
    #          's3://raster-vision-iowa-wind-turbines/images/m_4309457_ne_15_1_20170724.tif',
    #          's3://raster-vision-iowa-wind-turbines/images/m_4309361_se_15_1_20170820.tif',
    #          's3://raster-vision-iowa-wind-turbines/images/m_4309361_se_15_1_20170820.tif']
    scenes = []
    for uri in tiffs:
        raster_source = rv.RasterSourceConfig.builder(t2v.REMOTE_GEOTIFF_SOURCE) \
                                             .with_uri(uri) \
                                             .with_channel_order([0,1,2]) \
                                             .build()

        label_store = rv.LabelStoreConfig.builder(t2v.TILE_EMBEDDINGS_STORE) \
                                         .build()

        scene_id = os.path.splitext(os.path.basename(uri))[0]
        scene =  rv.SceneConfig.builder() \
                               .with_id(scene_id) \
                               .with_raster_source(raster_source) \
                               .with_label_store(label_store) \
                               .build()
        scenes.append(scene)
    return scenes

def get_predict_scenes(data_dir):
    pass
