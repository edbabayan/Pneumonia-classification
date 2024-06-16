from pathlib import Path


class CFG:
    root = Path(__file__).parent.parent.absolute()
    data = root.joinpath('data')
    image_size = 256
