- **v10-11**: No stars, but with padding (all galaxy types)

- **v9**: No stars and no padding

- **v6**: Trying no galaxies in padding to ensure this is what impacts training

- **v5**: In theory the one for the paper, corrected issue where there were no galaxies in the padding.

- **v4**: Uses `i < 27.3` as for final paper results. But no stars, and `mean_sources` = 4, which
    which should be adjusted. Total is galaxies is `1028 * 20`

- **v3**: corresponds to a dataset of only bright galaxies with total `512 * 20` in training + validation
