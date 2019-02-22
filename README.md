# tile2vec Raster Vision Plugin

This repo contains a plugin for Raster Vision that allows for usage of the [tile2vec](https://arxiv.org/abs/1805.02855) algorithm.

This codebase takes it's tile2vec PyTorch implementation from https://github.com/ermongroup/tile2vec

## Codebase Layout

###  tile2vec

The original code from https://github.com/ermongroup/tile2vec lives in the base package of `tile2vec`.
The code has been modified slightly to work with the Raster Vision plugin better.

One large change is that instead of baking triplets from imagery as a step, and then reading from
those triplets at train time, the `TileTripletDataset` reads from images in S3 directly.
The `TileTripletDataset` takes a set of Raster Vision scenes, and for each triplet,
randomly selects 2 scenes - one produces the anchor and neighbor of the triplet, and another
randomly selects a distant tile. This procedure assumes that two randomly selected windows
of two randomly selected scenes will fit the requirements of a 'distant' tile in the triplet.

### tile2vec.plugin

This subpackage contains the Raster Vision plugin, with the following components:

- `Tile2VecTask`: Defines a Task for tile2vec. It skips all the chipping steps, since the
chips are generated from imagery at train time as described above.
- `Tile2VecBackend`: This implants the training code from https://github.com/ermongroup/tile2vec
into the training method of Raster Vision, as well as implements prediction that produces embeddings.
- `TileEmbeddings`: This is a Raster Vision `Labels` implementation that maps chip windows
to numpy arrays that are the vector space embeddings produced by the Tile2Vec model.
- `TileEmbeddingsStore`: The `LabelStore` implementation for saving off tile embeddings.
- `RemoteRasterSource`: This is a custom raster source that allows for completely remote reading
of GeoTIFFs off of S3 - this functionality will be merged into Raster Vision as an option,
so this should be unecessary with a later Raster Vision release.
- `Tile2VecEvaluator`: This is a custom `Evaluator` that hijacks the evaluation step, usually
used to compare predictions to ground truth, to build up a [faiss](https://github.com/facebookresearch/faiss) index of embeddings
across scenes and chips, along with a csv that can  be used  as a lookup table of scenes and windows
of the index.

## Jupyter Notebooks

- `Display Triplets` allows you to pull triplets out of the TileTripletDataset and visualize them.
- `Search via embeddings` allows you to use the `faiss` index and lookup table to do a visual search on imagery.
