class NoteBookPlottingParams:
    fontsize = 20
    dpi = 300
    figsize = (8, 8)
    color_dict = {
        "bliss": "blue",
        "lsst": "orange",
        "detection_plot": {
            "only_bliss": "aqua",
            "only_lsst": "orange",
            "both": "lime",
            "neither": "red",
        },
        "classification_acc_bar_plot": {
            "bar_colors": ["slateblue", "darkslateblue", "blueviolet", "indigo"],
        },
        "detection_bar_plot": {
            "ground_truth": "green",
        },
        "flux_error_plot": {
            "band_colors": ["violet", "green", "red", "peru", "teal", "darkorange"],
        },
        "roc_plot": {
            "faint": "cornflowerblue",
            "bright": "blue",
        },
    }
