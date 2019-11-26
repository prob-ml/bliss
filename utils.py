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
