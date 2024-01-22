import os
import glob

from ml_data_set import MLDataSet


class DataLoader:
    @staticmethod
    def create_dataset(directory_path, column_titles, file_extension='wav'):
        """
        Create a dataset from audio files in a given directory using MLDataSet.

        This function reads audio files from a specified directory,
        parses their filenames to extract data based on the provided column titles,
        and returns an MLDataSet object containing this information.

        The function expects filenames to follow a specific pattern:
        <value1>_<value2>_<value3>...<ext>
        Each value corresponds to a column title provided in column_titles.
        For example: "actor-1_long-noise_small-size_light-weight.wav"

        Args:
        directory_path (str): Path to the directory containing the audio files.
        column_titles (list): List of column titles corresponding to values in the filename.
        file_extension (str): The file extension of the audio files (default is 'wav').

        Returns:
        MLDataSet: An object of MLDataSet containing the audio file paths and extracted data.
        """

        ml_dataset = MLDataSet()

        # Iterate over files in the directory
        for filepath in glob.glob(os.path.join(directory_path, f'*.{file_extension}')):
            filename = os.path.basename(filepath).split('.')[0]  # Remove the file extension

            # Extract values based on the filename and map them to the corresponding column titles
            values = filename.split('_')
            record = {column: value for column, value in zip(column_titles, values)}
            record['file_path'] = filepath

            # Add the record to the MLDataSet
            ml_dataset.add_sample(record)

        return ml_dataset
