# this is used exclusively for distributed training (with multiple GPUs)
class GlobalEnv:
    seed_in_this_program = None
    current_encoder_epoch = None
