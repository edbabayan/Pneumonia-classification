from pathlib import Path
import torch


class CFG:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    image_size = 256

    root = Path(__file__).parent.parent.absolute()
    data = root.joinpath('data')
    checkpoints = root.joinpath('checkpoints')
    processed_data = root.joinpath('processed')
    logs = root.joinpath('logs')

    batch_size = 32
    num_workers = 4

    model_weights = checkpoints.joinpath('epoch=44-step=33750.ckpt')
