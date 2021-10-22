from torch.optim import Adam, AdamW


def load_optimizer(model_params, hparams):
    assert hparams["optimizer_params"] is not None, "Need to specify 'optimizer_params'."
    name = hparams["optimizer_params"]["name"]
    kwargs = hparams["optimizer_params"]["kwargs"]
    return get_optimizer(name, model_params, kwargs)


def get_optimizer(name, model_params, optim_params):
    if name == "Adam":
        return Adam(model_params, **optim_params)

    if name == "AdamW":
        return AdamW(model_params, **optim_params)

    raise NotImplementedError(f"The requested optimizer '{name}' is not implemented.")
