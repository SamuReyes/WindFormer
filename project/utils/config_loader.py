import yaml

def load_config():
    with open("./config/params.yaml", "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)