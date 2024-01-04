import random

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class MLDataSet:
    def __init__(self):
        self.data = []

    def add_sample(self, sample):
        self.data.append(sample)

    def get_single_column(self, col_name):
        """Extracts and returns values from a single column."""
        return np.array([sample[col_name] for sample in self.data])

    def get_multiple_columns(self, col_names):
        """Extracts and returns values from multiple columns as a list of lists."""
        return np.array([[sample[col] for col in col_names] for sample in self.data])

    def split(self, train_size=0.7, val_size=0.15, test_size=0.15):
        assert train_size + val_size + test_size == 1, "The split sizes must sum up to 1"

        train, temp = train_test_split(self.data, train_size=train_size)
        val, test = train_test_split(temp, test_size=test_size / (val_size + test_size))

        return MLDataSet.from_list(train), MLDataSet.from_list(val), MLDataSet.from_list(test)

    def encode_categorical_columns(self, columns):
        label_encoders = {col: LabelEncoder() for col in columns}

        for col in columns:
            column_data = [record[col] for record in self.data]
            encoded_data = label_encoders[col].fit_transform(column_data)

            # Update the dataset with encoded data
            for record, encoded_val in zip(self.data, encoded_data):
                record[col] = encoded_val

        # Optionally, you can return the label encoders if you need to reverse the encoding later
        return label_encoders

    def remove(self, condition):
        """
        Remove samples from the dataset based on a given condition.

        Args:
        condition (callable): A function that takes a sample as input and returns True
                              if the sample should be removed, False otherwise.
        """
        self.data = [sample for sample in self.data if not condition(sample)]

    @staticmethod
    def from_list(data_list):
        dataset = MLDataSet()
        dataset.data = data_list
        return dataset

# # Example usage
# dataset = MLDataSet()
# dataset.add_sample({'feature1': 1, 'feature2': 2, 'label': 'A'})
# dataset.add_sample({'feature1': 3, 'feature2': 4, 'label': 'B'})
# # ... add more samples ...
#
# X = dataset.get_features(['feature1', 'feature2'])
# y = dataset.get_labels(['label'])
#
# train_set, val_set, test_set = dataset.split(train_size=0.7, val_size=0.15, test_size=0.15)
