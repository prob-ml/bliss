import warnings


def deprecated(deprecated_in_version, message):
    """Decorator to mark a function as deprecated."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            warnings.warn(
                "Call to deprecated function {0}.".format(func.__name__)
                + " Deprecated since version {0}.".format(deprecated_in_version)
                + " {0}".format(message),
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator
