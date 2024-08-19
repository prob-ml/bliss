class NoteBookPlottingParams:
    fontsize = 8
    dpi = 1000
    figsize = (3.4, 3.4)
    color_dict = {
        "bliss": "blue",
        "lsst": "orange",
        "detection_plot": {
            "only_bliss": "blue",
            "only_lsst": "orange",
            "both": "lime",
            "neither": "red",
        },
        "classification_acc_bar_plot": {
            "bar_colors": ["green", "red", "purple", "brown"],
        },
        "detection_bar_plot": {
            "ground_truth": "green",
        },
        "flux_error_plot": {
            "band_colors": ["violet", "green", "red", "peru", "teal", "darkorange"],
        },
        "roc_plot": {
            "faint": "red",
            "bright": "blue",
        },
    }
