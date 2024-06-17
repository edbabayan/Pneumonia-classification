from pathlib import Path
import torch


class CFG:
    root = Path(__file__).parent.parent.absolute()
    data = root.joinpath('data')
    checkpoints = root.joinpath('checkpoints')
    processed_data = root.joinpath('processed')
    logs = root.joinpath('logs')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_size = 256
