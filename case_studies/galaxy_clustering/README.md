# Galaxy clustering
*Winter 2024*

*Li Shihang, Zhao Yongxiang, [Gabriel Patron](https://lsa.umich.edu/stats/people/phd-students/gapatron.html), [Prof. Jeffrey Regier](https://regier.stat.lsa.umich.edu/).*

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


### URPS
This project is being conducted under the Undergraduate Research Program in Statistics (URPS), a competitive program that pairs promising undergraduates with Statistics faculty on a research project for the winter semester. If you are interested, and for more information, please follow [this link](https://lsa.umich.edu/stats/undergraduate-students/undergraduate-research-opportunities-.html).
