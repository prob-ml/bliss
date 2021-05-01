from torch.optim import Adam, AdamW


def get_optimizer(name, model_params, optim_params):
    if name == "Adam":
        return Adam(model_params, **optim_params)

    if name == "AdamW":
        return AdamW(model_params, **optim_params)

    raise NotImplementedError(f"The requested optimizer '{name}' is not implemented.")
