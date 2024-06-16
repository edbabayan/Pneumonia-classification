import pandas as pd
import numpy as np
import cv2
import pydicom
from tqdm import tqdm

from preprocess.preprocess_config import CFG


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.labels = pd.read_csv(self.config.train_labels_csv).drop_duplicates(subset='patientId')
        self.sums = 0
        self.sums_squared = 0

    def process(self):
        for c, patient in enumerate(tqdm(self.labels['patientId'], desc='Processing images')):
            patient_id = self.labels.patientId.iloc[c]
            dcm_path = self.config.train_images.joinpath(patient_id).with_suffix('.dcm')
            dcm = pydicom.read_file(dcm_path).pixel_array / 255

            dcm_array = cv2.resize(dcm, (self.config.image_size, self.config.image_size)).astype(np.float16)

            label = self.labels.Target.iloc[c]
            train_or_valid = 'train' if c < 0.8 * len(self.labels) else 'valid'

            current_save_path = self.config.output_path.joinpath(train_or_valid).joinpath(str(label))
            current_save_path.mkdir(parents=True, exist_ok=True)
            np.save(current_save_path.joinpath(patient_id), dcm_array)

            normalizer = 224 * 224
            if train_or_valid == 'train':
                self.sums += np.sum(dcm_array) / normalizer
                self.sums_squared += np.sum(dcm_array ** 2) / normalizer

    def get_statistics(self):
        mean = self.sums / len(self.labels) * 0.8
        std = np.sqrt(self.sums_squared / len(self.labels) * 0.8 - mean ** 2)
        return mean, std


if __name__ == '__main__':
    process = Preprocessor(CFG)
    process.process()
    _mean, _std = process.get_statistics()
    print(_mean, _std)
