import os
from PIL import Image
import pydicom
import nibabel as nib


class InputProcessor:
    """
    A class to process input files for a pneumonia detection model.
    """

    def __init__(self):
        # Define the supported file types
        self.supported_file_types = ['jpg', 'jpeg', 'png', 'dcm', 'ima', 'nii', 'nii.gz']

        # Map common extensions to a readable format
        self.extension_map = {
            '.jpg': 'JPEG image',
            '.jpeg': 'JPEG image',
            '.png': 'PNG image',
            '.dcm': 'DICOM file',
            '.ima': 'IMA file',
            '.nii': 'NIfTI file',
            '.nii.gz': 'NIfTI file'
        }

    def get_file_type(self, file_path):
        """
        Method to determine the file type based on its extension.

        Parameters:
        file_path (str): The path to the input file.

        Returns:
        str: The type of file based on its extension.
        """
        # Extract the file extension
        file_extension = os.path.splitext(file_path)[1].lower()

        # Special case for NIfTI files with .nii.gz extension
        if file_extension == '.gz' and file_path.endswith('.nii.gz'):
            file_extension = '.nii.gz'

        # Check if the file extension is supported
        if file_extension in self.extension_map:
            file_type = self.extension_map[file_extension]
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        return file_type

    def load_image(self, file_path):
        file_type = self.get_file_type(file_path)

        if file_type in ['JPEG image', 'PNG image']:
            return Image.open(file_path).convert('L')
        elif file_type == 'DICOM file':
            dicom = pydicom.dcmread(file_path)
            return Image.fromarray(dicom.pixel_array).convert('L')
        elif file_type in ['NIfTI file']:
            nifti = nib.load(file_path)
            data = nifti.get_fdata()
            slice_2d = data[:, :, data.shape[2] // 2]  # Take the middle slice
            return Image.fromarray(slice_2d).convert('L')
        else:
            raise ValueError(f"Unsupported file type: {file_type}")