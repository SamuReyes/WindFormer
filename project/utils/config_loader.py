import yaml


def load_config():
    """
    Loads configuration settings from a YAML file.

    :return: A dictionary containing the configuration parameters.
    """
    with open("./config/params.yaml", "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)
