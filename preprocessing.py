import cv2
import pydicom
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from config import CFG

labels = pd.read_csv(CFG.train_labels_csv)

labels = labels.drop_duplicates(subset='patientId')

sums, sums_squared = 0, 0


for c, patient in enumerate(tqdm(labels['patientId'])):
    patient_id = labels.patientId.iloc[c]
    dcm_path = CFG.train_images.joinpath(patient_id)
    dcm_path = dcm_path.with_suffix('.dcm')
    dcm = pydicom.read_file(dcm_path).pixel_array / 255

    dcm_array = cv2.resize(dcm, (CFG.image_size, CFG.image_size)).astype(np.float16)

    label = labels.Target.iloc[c]

    train_or_valid = 'train' if c < 0.8 * len(labels) else 'valid'

    current_save_path = CFG.output_path.joinpath(train_or_valid).joinpath(str(label))
    current_save_path.mkdir(parents=True, exist_ok=True)
    np.save(current_save_path.joinpath(patient_id), dcm_array)

    normalizer = 224 * 224
    if train_or_valid == 'train':
        sums += np.sum(dcm_array) / normalizer
        sums_squared += np.sum(dcm_array ** 2) / normalizer

print('')