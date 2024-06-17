import pandas as pd
import numpy as np
import cv2
import pydicom
from tqdm import tqdm

from preprocess.preprocess_config import CFG


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.labels = pd.read_csv(self.config.train_labels_csv).drop_duplicates("patientId")
        self.sums = 0
        self.sums_squared = 0

    def process(self):
        for c, patient_id in enumerate(tqdm(self.labels.patientId)):
            dcm_path = CFG.train_images / patient_id
            dcm_path = dcm_path.with_suffix(".dcm")

            dcm = pydicom.read_file(dcm_path).pixel_array / 255

            dcm_array = cv2.resize(dcm, (224, 224)).astype(np.float16)

            label = self.labels.Target.iloc[c]

            train_or_val = "train" if c < 24000 else "val"

            current_save_path = CFG.output_path / train_or_val / str(label)
            current_save_path.mkdir(parents=True, exist_ok=True)
            np.save(current_save_path / patient_id, dcm_array)

            normalizer = dcm_array.shape[0] * dcm_array.shape[1]
            if train_or_val == "train":
                self.sums += np.sum(dcm_array) / normalizer
                self.sums_squared += (np.power(dcm_array, 2).sum()) / normalizer

    def get_statistics(self):
        mean = self.sums / 24000
        std = np.sqrt(self.sums_squared / 24000 - (mean ** 2))
        return mean, std


if __name__ == '__main__':
    process = Preprocessor(CFG)
    process.process()
    _mean, _std = process.get_statistics()
    print(_mean, _std)
