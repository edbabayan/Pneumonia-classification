from pathlib import Path
import cv2
import pydicom
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


labels = pd.read_csv('data/stage_2_train_labels.csv')

labels = labels.drop_duplicates(subset='patientId')

ROOT_PATH = Path("stage_2_train_images/")
OUTPUT_PATH = Path("processed")

sums, sums_squared = 0, 0



print('')