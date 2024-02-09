import yaml


def load_config():
    """
    Loads configuration settings from a YAML file.

    :return: A dictionary containing the configuration parameters.
    """
    with open("./config/params.yaml", "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def assign_config(wandb_config, config, parent_key='', target_keys=['preprocessing', 'model', 'train']):
    """
    Assigns configuration settings to the wandb_config object.

    :param wandb_config: The wandb.config object.
    :param config: The configuration settings.
    :param parent_key: The parent key.
    :param target_keys: The keys to assign.
    """
    for key, value in config.items():
        if key in target_keys or parent_key in target_keys:
            full_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
                assign_config(
                    wandb_config, value, full_key, target_keys)
            else:
                setattr(wandb_config, str(key), value)
        elif isinstance(value, dict):
            assign_config(wandb_config, value, key, target_keys)
