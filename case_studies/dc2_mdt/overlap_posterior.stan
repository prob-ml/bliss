data {
    int<lower=1> num_objects;
    int<lower=1> img_size;
    array[img_size, img_size] int<lower=0> img;
    array[num_objects] matrix[img_size, img_size] psf;
    real<lower=0.0> flux_alpha;
    real<lower=0.0> flux_beta;
    real<lower=0.0> background_intensity;
}

parameters {
    vector<lower=0.01, upper=10000.0>[num_objects] fluxes;
}

model {
    fluxes ~ gamma(flux_alpha, flux_beta);
    matrix[img_size, img_size] intensities = rep_matrix(0.0, img_size, img_size);
    for (i in 1:num_objects) {
        intensities = intensities + psf[i] * fluxes[i];
    }
    intensities = intensities + background_intensity;
    for (i in 1:img_size) {
        for (j in 1:img_size) {
            img[i, j] ~ poisson(intensities[i, j]);
        }
    }
}
