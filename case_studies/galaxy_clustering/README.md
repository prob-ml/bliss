# Galaxy clustering
*Winter 2024*

*[Li Shihang](https://www.linkedin.com/in/shihang-li-2b69251ba/), Zhao Yongxiang, [Gabriel Patron](https://lsa.umich.edu/stats/people/phd-students/gapatron.html), [Prof. Jeffrey Regier](https://regier.stat.lsa.umich.edu/).*

----------------------------------------------------------------------------------------------------------------------

Galaxy clusters, which are made up of hundreds or thousands of nearby galaxies, are the largest structures in the universe. Yet finding them in astronomical images is challenging: galaxies often appear close together in 2d images without being part of a common cluster. In this project, we take a probabilistic approach to finding and characterizing galaxy clusters. To perform posterior inference, we’ll use a new technique called neural posterior estimation, which involves simulating images with and without galaxy clusters, and training a convolutional neural network (CNN) to predict whether each galaxy is part of a cluster and if so, to predict the cluster’s properties. Undergraduate researchers will 1) use Python to implement and compare several galaxy cluster simulators, 2) use PyTorch to train a neural network to predict the locations and masses of galaxy clusters in simulated images, and 3) apply the trained neural network to real astronomical images.


### Goals
We set the following, non-exhaustive, objectives for our galaxy cluster detection project.
- **Generative Model Development** Define, implement, refine the generative model by subclassing 'CatalogPrior' and simulate realistic galaxy cluster data.
- **Benchmarking Metrics**: Adapt the 'Metric' class to  galaxy cluster detection benchmarks.
- **Inference Model Specification**: Outline variational distributions by subclassing 'VariationalDistSpec' and 'VariationalDist' for probabilistic modeling.
- **Training BLISS**: Harness BLISS to train our inference model and fine-tune it for optimal performance. 

### References
- [Variational Inference for Deblending Crowded Starfields](https://arxiv.org/pdf/2102.02409.pdf).
- [Euclid preparation III](https://arxiv.org/pdf/1906.04707.pdf)
- [YOLO-CL](https://arxiv.org/pdf/2301.09657.pdf).
- [redMaPPer I](https://arxiv.org/abs/1303.3562)
- [redMaPPer IV](https://arxiv.org/abs/1410.1193).
- [redMaPPer (latest)](https://arxiv.org/abs/1601.00621).
- [LSST Science Book V2 Chapter 9.5.3](https://www.lsst.org/sites/default/files/docs/sciencebook/SB_9.pdf).
- [LSST Science Book V2 Chapter 12.12](https://www.lsst.org/sites/default/files/docs/sciencebook/SB_12.pdf).
- [LSST Science Book V2 Chapter 13.6](https://www.lsst.org/sites/default/files/docs/sciencebook/SB_13.pdf).
- [LSST Science Book V2 Chapter 14.3.8](https://www.lsst.org/sites/default/files/docs/sciencebook/SB_14.pdf).


### URPS
This project is being conducted under the Undergraduate Research Program in Statistics (URPS), a competitive program that pairs promising undergraduates with Statistics faculty on a research project for the winter semester. If you are interested, and for more information, please follow [this link](https://lsa.umich.edu/stats/undergraduate-students/undergraduate-research-opportunities-.html).

### Documentation

#### Generative Model

##### FullCatalog vs. TileCatalog
Current implementaion focueses on building from FullCatalog instead of TileCatalog.
First dimension is always batch size (# of images). Second dimension is usually the maximum number of sources within the batch to have an upper bond for consistent tensor shape. The last dimension contains info for the specific source at specific batch.  
**Example:** A tensor shape of (32, 1500, 2) means we have 32 images in a batch, the maximum possible number of sources across whole batch is 1500 and, for each source, it will have two properties in like (x, y).  


##### Details
Our implementaion's goal is to create a single cluster (in the future, 0-k) within the image. The cluster consists mainly of galaxies. Therefore, the *galaxy_fluxes*, *galaxy_params*, and density of galaxies within the cluster should perform differently than the other part.

*The task may be seen as creating a subimage inside the original image.* We consider a cluster with a center. For the center, it would have a bounding box and all galaxies within the bounding box should perform differently. To make the image look more real, the center should be in another bounding box of the overall image encompassing the region between 25% and 75% of its side length. The center's own bounding box shall have side length no larger than 50% of the image side length. 

In our cluster, we first calcuate the area of the bounding box and convert it to the equal number of titles. Then we times it with maximum number of sources within a tile and randomly pick 80% out of it (gives a slightly larger value than the average number of sources per tile). The fluxes/params for the galaxy so far follows the identical implementation for normal galaxies. In the future, we aim to use the redmapper paper to further modify the flux and params. 

So far, the only change within the cluster was to increase the rate of galaxies. That is, we use the Possion distribution to first sample the number of sources based on number of tiles and the average number of sources within tiles. A predetermined probability of a source being a galaxy determines how many galaxies will ultimately be present within the image. 

For implementation details, we generate location, fluxes, params, and types for the cluster first. Then we follow the normal procedure of generating sources. In the end, we stack two sets of vectors together to have the final values. 

**Returns:**
1. *n_sources*: in shape (batch_size, ). A single integer representing how many sources in each image.
2. *source_type*: in shape (batch_size, max(n_sources), 1). Uses integer to represent source type. 0 represents a star while 1 represents galaxy. Second dimension is adjusted to the maximum number of sources within the batch.
3. *plocs*: in shape (batch_size, max(n_sources), 2). The coordinates for sources are in form (x, y) and represents absolute coordinates.
4. *galaxy_fluxes/star_fluxes/galaxy_params*:  in shape (batch_size, max(n_sources), n_bands).

