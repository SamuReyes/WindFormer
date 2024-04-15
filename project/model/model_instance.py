import torch
from model.model import ViViT

gpu = 0 # Change according GPU you want to use (0, 1, ...)
device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')


def init_model(config):
    """
    Initializes the ViViT model based on the provided configuration.
    """

    # Extract model and training parameters from the config
    image_size_3d = config['model']['image_size_3d']
    patch_size_3d = config['model']['patch_size_3d']
    image_size_2d = config['model']['image_size_2d']
    patch_size_2d = config['model']['patch_size_2d']
    dim = config['model']['dim']
    depth = config['model']['depth']
    heads = config['model']['heads']
    dim_head = config['model']['dim_head']
    dropout = config['model']['dropout']
    emb_dropout = config['model']['emb_dropout']
    reconstr_dropout = config['model']['reconstr_dropout']
    scale_dim = config['model']['scale_dim']
    sequence_length = config['train']['sequence_length']

    # Initialize the model and move it to the configured device
    model = ViViT(image_size_3d, patch_size_3d, image_size_2d, patch_size_2d, sequence_length,
                  dim, depth, heads, dim_head, dropout, emb_dropout, reconstr_dropout, scale_dim).to(device)

    return model
