import os

OFF_MSG = "Set os.environ['CANDLE_WARNINGS'] = 'OFF' to disable warnings."


def beta(func):
    """
    Decorator to indicate a beta feature.
    """

    def wrapper(*args, **kwargs):
        if os.environ.get("CANDLE_WARNINGS", "ON") == "ON":
            print(
                f"Warning: The function '{func.__name__}' is a beta feature and may not work properly!" + "\n" + OFF_MSG)
        return func(*args, **kwargs)

    return wrapper


def deprecated(func):
    """
    Decorator to indicate a deprecated function.
    """

    def wrapper(*args, **kwargs):
        if os.environ.get("CANDLE_WARNINGS", "ON") == "ON":
            print(f"Warning: The function '{func.__name__}' is deprecated and may be removed in future versions!" + "\n" + OFF_MSG)
        return func(*args, **kwargs)

    return wrapper


def experimental(func):
    """
    Decorator to indicate an experimental feature.
    """

    def wrapper(*args, **kwargs):
        if os.environ.get("CANDLE_WARNINGS", "ON") == "ON":
            print(f"Warning: The function '{func.__name__}' is experimental and could behave unexpectedly!" + "\n" + OFF_MSG)
        return func(*args, **kwargs)

    return wrapper


def custom_warning(message):
    """
    Function to display a custom warning message.
    """
    if os.environ.get("CANDLE_WARNINGS", "ON") == "ON":
        print(f"Warning: {message}" + "\n" + OFF_MSG)
