from pathlib import Path


class CFG:
    root = Path(__file__).parent.absolute()
    data = root.joinpath('data')
    train_labels_csv = data.joinpath('stage_2_train_labels.csv')
    train_images = data.joinpath('stage_2_train_images')
    output_path = root.joinpath('processed')
    image_size = 256
