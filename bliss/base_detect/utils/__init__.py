# flake8: noqa
import platform


def emojis(str=""):  # pylint: disable=W0622
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode("ascii", "ignore") if platform.system() == "Windows" else str
