#import pytest
#import json

#from celeste.utils import const
#from celeste.data import simulated_datasets_lib
#from celeste.models import sourcenet_lib
#from celeste import run_sleep_galaxy


# TODO: Set up pytest.mark.slow to enble skip on slow test


#def pytest_addoption(parser):
    """
    Add --runslow option on command line for this project
    """
#    parser.addoption(
#        "--runslow", action="store_true", default=False, help="run slow tests"
#    )


#def pytest_configure(config):
    """
    Add slow markers to pytest
    """
#    config.addinivalue_line("markers", "slow: mark test as slow to run")


#def pytest_collection_modifyitems(config, items):
#    if config.getoption("--runslow"):
#        # --runslow given in cli: do not skip slow tests
#        return
#    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
#    for item in items:
#        if "slow" in item.keywords:
#            item.add_marker(skip_slow)


# TODO: Create a 40*40 test galaxy image and save it into data folder
#       as well as the parameters
# Question: Which data folder?

#param_file = const.data_path.joinpath("default_galaxy_parameters.json")
#with open(param_file, "r") as fp:
#    data_params = json.load(fp)

#data_params["slen"] = 40
#data_params["max_galaxies"] = 2

#galaxy_test_dataset = simulated_datasets_lib.GalaxyDataset.load_dataset_from_params(
#    n_images=1, data_params=data_params,
#)

# Questions: Are they all source parameters?
#(
#    n_sources,
#    locs,
#    gal_params,
#    single_galaxies,
#) = galaxy_test_dataset.simulator.sample_parameters(batchsize=1)

# get image
# Questions: Does this return a tensor?
#galaxy_test_image = galaxy_test_dataset.simulator.generate_images(
#    locs=locs, n_sources=n_sources, galaxies=single_galaxies
#).to(const.device)


#class TestGalaxySleepEncoder:
    # TODO: Use run_sleep_trian to train the Galaxy encoder in
    #       sleep-face
    # need to set global device
#    @pytest.mark.slow
#    def test_galaxy_sleep(self, tmp_path):
        # tmp_path is equivalent to pathlib.Path
#        state_dict_file = tmp_path.mkdir("pytest_temp").join("galaxy_i.dat")
#       output_file = tmp_path.mkdir("pytest_temp").join("output.txt")

        # setup train dataset
        # Questions: What to use for n_image?
#        n_image = 100
#        data_params["max_galaxies"] = 20
#        data_params["mean_galaxies"] = 10
#        galaxy_train_dataset = simulated_datasets_lib.GalaxyDataset.load_dataset_from_params(
#            n_images=n_image, data_params=data_params
#        )

        # setup encoder
        # train encoder on 100*100 images
#        galaxy_encoder = sourcenet_lib.SourceEncoder(
#            slen=100,
#            n_bands=1,
#            ptile_slen=20,
#            step=5,
#            edge_padding=5,
#            max_detections=2,
#            n_source_params=galaxy_train_dataset.simulator.latent_dim,
#        ).to(const.device)

        # set up optimizer and run training
#        optimizer = run_sleep_galaxy.get_optimizer(galaxy_encoder)
#        run_sleep_galaxy.train(
#            galaxy_encoder,
#            galaxy_train_dataset,
#            optimizer,
#            state_dict_file,
#            output_file,
#        )

        # TODO: obtain estimates of locations and score them
        # Questions: not sure how to use sample_encoder to achieve this since
        #            I'm not clear about what should be the parameter image
        #            and training
        #
        # Questions: Also, I'm not sure how to proceed scoring. Should I use
        #            one of the loss function and test a threshold in assertion
        #            or just draw images with checker?
