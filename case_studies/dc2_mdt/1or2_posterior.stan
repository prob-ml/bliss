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
    simplex[num_objects] theta;
}

model {
    fluxes ~ gamma(flux_alpha, flux_beta);
    matrix[img_size, img_size] intensities = rep_matrix(0.0, img_size, img_size);
    theta ~ dirichlet(rep_vector(1.0, num_objects));
    vector[num_objects] log_lik = log(theta);
    // vector[num_objects] log_lik = log(rep_vector(1.0 / num_objects, num_objects));
    for (i in 1:num_objects) {
        intensities = intensities + psf[i] * fluxes[i];
        for (img_i in 1:img_size) {
            for (img_j in 1:img_size) {
                log_lik[i] += poisson_lpmf(img[img_i, img_j] | intensities[img_i, img_j] + background_intensity);
            }
        }
    }
    target += log_sum_exp(log_lik);
}
