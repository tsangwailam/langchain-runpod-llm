from importlib.metadata import version, PackageNotFoundError


def get_version():
    """ Get the version of langchain-runpod-llm. """
    try:
        return version("langchain-runpod-llm")
    except PackageNotFoundError:
        return "unknown"


__version__ = get_version()
