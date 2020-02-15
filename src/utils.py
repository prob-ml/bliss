def to_argparse_form(s):
    """
    Turns a string s into the form that is required by argparser.
    :param s: A string.
    :return: A string
    """
    new_s = '--'
    for c in s:
        if c == '_':
            new_s += '-'
        else:
            new_s += c
    return new_s


def str_bool(s):
    if s.lower() in ['yes', 'true', '1']:
        return True
    elif s.lower() in ['no', 'false', '0']:
        return False
    else:
        raise ValueError("Invalid value for str_bool.")


general_args = [
    'device',
    'dir_name',
    'overwrite',
    'seed',
    'nocuda',
    'model',
    'dataset',
    'batch_size',
    'training_examples',
    'evaluation_examples',
    'evaluate',
    'epochs',
]
