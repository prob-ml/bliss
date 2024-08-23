# Inference on Galaxy Cluster Fields

Summer 2024

[Ishan Kapnadak](https://www.linkedin.com/in/ishan-kapnadak/), [Gabriel Patron](https://lsa.umich.edu/stats/people/phd-students/gapatron.html), [Prof. Jeffrey Regier](https://regier.stat.lsa.umich.edu/), [Prof. Camille Avestruz](https://cavestruz.github.io/).

Built on work done in Winter 2024 by [Li Shihang](https://www.linkedin.com/in/shihang-li-2b69251ba/), Zhao Yongxiang.

----------------------------------------------------------------------------------------------------------------------

## Generation of Data

The data generation routine proceeds through phases. The entire routine is conveniently wrapped into a single python script `data_gen.py` that draws its parameters from the Hydra configuration, located under `conf/config.yaml` under the `data_gen` key. These phases proceed as follows.

1. **Catalog Generation.** First, we sample semi-synthetic source catalogs with their relevant properties, which are stored as `.dat` files in the `data_dir/catalogs` subdirectory.
2. **Image Generation.** Then, we take in the aforementioned source catalogs and use GalSim to render them as images, which are stored as `.fits` files (one for each band) in the `data_dir/images` subdirectory.
3. **File Datum Generation.** Finally, we convert the full source catalogs generated in phase 1 into tile catalogs, stack them up with their corresponding images, and store these objects as `.pt` files (which is what the encoder ultimately uses) in the `data_dir/file_data` subdirectory.

The following parameters can be set within the configuration file `config.yaml`.
1. `data_dir`: the path of the directory where generated data will be stored.
2. `image_size`: size of the image (pixels).
3. `tile_size`: size of tile to be used (pixels).
4. `nfiles`: number of files to be generated.
5. `n_catalogs_per_file`: number of catalogs to be stored in each file datum object.
6. `bands`: survey bands to be used (`["g", "r", "i", "z"]` for DES).
7. `min_flux_for_loss`: minimum flux for filtering.
8. `overwrite`: whether to overwrite existing files (only works for catalogs for now)
