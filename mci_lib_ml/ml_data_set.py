import base64
import json
import pickle
import random

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from mci_lib_ml.basic_operations import Ops


class MLDataSet:
    def __init__(self):
        self.data = []
        self.encoders = {}

    @staticmethod
    def from_list(data_list):
        dataset = MLDataSet()
        dataset.data = data_list
        return dataset

    def add_sample(self, sample):
        self.data.append(sample)

    def get_columns(self, col_names):
        """Extracts and returns a new MLDataSet with values from multiple columns."""
        new_dataset = MLDataSet()
        for sample in self.data:
            new_sample = {col: sample[col] for col in col_names}
            new_dataset.add_sample(new_sample)
        return new_dataset

    def get_feature_length(self, columns):
        total_length = 0
        for column in columns:
            # Check if the column exists in the data
            if column in self.data[0]:
                total_length += len(self.data[0][column])
            else:
                print(f"Warning: Column '{column}' not found in data.")
        return total_length

    def split(self, train_size=0.7, val_size=0.15, test_size=0.15, stratify_cols=None):
        assert train_size + val_size + test_size == 1, "The split sizes must sum up to 1"

        stratify_labels = None
        if stratify_cols is not None:
            stratify_labels = ['_'.join(str(sample[col]) for col in stratify_cols) for sample in self.data]

        # First split: Train and Temp (Val + Test)
        train, temp = train_test_split(self.data, train_size=train_size, stratify=stratify_labels)

        # Create new stratification labels for the second split
        stratify_labels_temp = None
        if stratify_cols is not None:
            stratify_labels_temp = ['_'.join(str(sample[col]) for col in stratify_cols) for sample in temp]

        # Adjust test size for the second split
        test_size_adjusted = test_size / (test_size + val_size)

        # Second split: Val and Test
        val, test = train_test_split(temp, test_size=test_size_adjusted, stratify=stratify_labels_temp)

        return MLDataSet.from_list(train), MLDataSet.from_list(val), MLDataSet.from_list(test)

    def encode_categorical_columns(self, columns):
        encoders = {}
        label_encoders = {col: LabelEncoder() for col in columns}

        for col in columns:
            if not col in self.data[0]:
                continue

            column_data = [record[col] for record in self.data]
            encoded_data = label_encoders[col].fit_transform(column_data)

            # Update the dataset with encoded data
            for record, encoded_val in zip(self.data, encoded_data):
                record[col] = encoded_val

            encoders[col] = label_encoders[col]

        self.encoders = self.encoders_to_dict(encoders)
        return self.encoders

    def decode_column(self, column_name):
        encoders = self.dict_to_encoders(self.encoders)

        if column_name not in encoders:
            raise ValueError(f"Encoder for column '{column_name}' not found.")

        label_encoder = encoders[column_name]
        encoded_data = [record[column_name] for record in self.data]
        decoded_data = label_encoder.inverse_transform(encoded_data)

        # Update the dataset with decoded data
        for record, decoded_val in zip(self.data, decoded_data):
            record[column_name] = decoded_val

    def encoders_to_dict(self, encoders):
        encoders_dict = {}
        for col, encoder in encoders.items():
            # Create a mapping of class to integer for each encoder
            class_to_int = {cls: int_val for cls, int_val in zip(encoder.classes_, range(len(encoder.classes_)))}
            encoders_dict[col] = class_to_int
        return encoders_dict

    def dict_to_encoders(self, encoders_dict):
        encoders = {}
        for col, class_to_int in encoders_dict.items():
            # Convert class_to_int mapping to sorted classes list
            classes = sorted(class_to_int, key=lambda cls: class_to_int[cls])
            encoder = LabelEncoder()
            encoder.classes_ = np.array(classes)
            encoders[col] = encoder
        return encoders

    def normalize_mean_and_variance(self, column):
        self.transform_column(column, lambda data: Ops.normalize_mean_and_variance(data))

    def transform_column(self, column, transformation_function):
        transformed_dataset = MLDataSet()

        for record in self.data:
            transformed_record = record.copy()
            transformed_record[column] = transformation_function(record[column])
            transformed_dataset.add_sample(transformed_record)

        self.data = transformed_dataset.data

    def extract_new_column(self, new_column, transformation_function):
        transformed_dataset = MLDataSet()

        for record in self.data:
            transformed_record = record.copy()
            transformed_record[new_column] = transformation_function(transformed_record)
            transformed_dataset.add_sample(transformed_record)

        self.data = transformed_dataset.data

    def augment_column(self,
                       column_to_augment,
                       label_for_augmented_samples,
                       augmentation_function):
        # Create a list to store augmented samples
        augmented_samples = []

        # Iterate over each sample in the dataset
        for record in self.data:
            # Initialize 'augmentations' column for original samples
            if 'augmentations' not in record:
                record['augmentations'] = []

            # Create an augmented copy of the sample
            augmented_record = record.copy()
            augmented_record[column_to_augment] = augmentation_function(augmented_record)

            # Add label for augmented samples to 'augmentations' column
            augmented_record['augmentations'] = record['augmentations'] + [label_for_augmented_samples]

            # Append the augmented sample to the list
            augmented_samples.append(augmented_record)

        # Extend the dataset with the augmented samples
        self.data.extend(augmented_samples)

    def remove(self, condition):
        """
        Remove samples from the dataset based on a given condition.

        Args:
        condition (callable): A function that takes a sample as input and returns True
                              if the sample should be removed, False otherwise.
        """
        self.data = [sample for sample in self.data if not condition(sample)]

    def validate_column_shape(self, column_name):
        """Checks if the data in the specified column has a consistent shape."""
        # Get the shape of the first item in the column
        first_shape = None
        for sample in self.data:
            if isinstance(sample[column_name], np.ndarray):
                first_shape = sample[column_name].shape
                break

        # Check if all other items have the same shape
        for sample in self.data:
            if isinstance(sample[column_name], np.ndarray) and sample[column_name].shape != first_shape:
                raise "Column does not have a consistent shape."

    def validate_multiple_column_shapes(self, columns):
        for column in columns:
            self.validate_column_shape(column)

    def fix_column_shape_by_truncation(self, column_name):
        """Truncates the data in the specified column to the minimum shape."""
        # Find the minimum shape for the specified column
        column_data = [sample[column_name] for sample in self.data if isinstance(sample[column_name], np.ndarray)]
        shapes = [item.shape for item in column_data]
        min_shape = tuple(min(s[i] for s in shapes) for i in range(len(shapes[0])))

        # Truncate data in the specified column to the minimum shape
        for sample in self.data:
            if isinstance(sample[column_name], np.ndarray):
                sample[column_name] = self.truncate_to_shape(sample[column_name], min_shape)

    def truncate_column_to_shape(self, column_name, shape):
        """Truncates the data in the specified column to a given shape."""
        # Truncate data in the specified column to the provided shape
        for sample in self.data:
            if isinstance(sample[column_name], np.ndarray):
                array = sample[column_name]

                # Ensure the target shape is smaller or equal to the array's shape
                array_shape = array.shape
                if len(shape) != len(array_shape):
                    raise ValueError("Target shape must have the same number of dimensions as the array's shape.")

                # Truncate each dimension as necessary
                slices = tuple(slice(0, min(array_shape[dim], shape[dim])) for dim in range(len(array_shape)))
                sample[column_name] = array[slices]

    def truncate_column_to_divisible_shape(self, column_name, divisible_by):
        """Truncates the data in the specified column to the largest shape evenly divisible by a specified number."""
        # Find the first sample with the specified column to determine the array shape
        for sample in self.data:
            if isinstance(sample[column_name], np.ndarray):
                array_shape = sample[column_name].shape
                break
        else:
            raise ValueError("No array found in the specified column.")

        # Calculate the target shape based on divisibility
        target_shape = tuple((dim_size // divisible_by) * divisible_by for dim_size in array_shape)

        # Call the original function to perform the truncation
        self.truncate_column_to_shape(column_name, target_shape)

    def truncate_multiple_columns_to_shape(self, column_shapes):
        """Truncates multiple columns to their respective shapes."""
        for column_name, shape in column_shapes.items():
            self.truncate_column_to_shape(column_name, shape)

    def truncate_to_shape(self, array, target_shape):
        """Truncates an array to the target shape."""


    def fix_multiple_columns_by_truncation(self, columns_to_fix):
        """Truncates the data in multiple specified columns to their minimum shapes."""
        for column in columns_to_fix:
            self.fix_column_shape_by_truncation(column)

    def truncate_to_shape(self, data, min_shape):
        """Truncates data to the specified minimum shape."""
        slices = tuple(slice(0, min_length) for min_length in min_shape)
        return data[slices]

    def get_shape(self):
        """Returns the shape of the dataset as (number of samples, number of columns)."""
        num_samples = len(self.data)
        num_columns = len(self.data[0].keys())
        return num_samples, num_columns

    def get_column_shape(self, column_name):
        """Returns the shape of a specified column."""
        # Collect shapes of the specified column
        column_shapes = [sample[column_name].shape for sample in self.data]

        # Determine the most representative shape (e.g., the largest shape)
        max_shape = max(column_shapes, key=lambda x: len(x))
        return max_shape

    def get_all_column_shapes(self):
        input_shapes = {}
        sample_keys = self.data[0].keys()

        for key in sample_keys:
            input_shapes[key] = self.get_column_shape(key)

        return input_shapes

    def to_numpy_arrays(self):
        """Converts the dataset into a list of NumPy arrays, each for a different column."""
        # Get the list of keys (columns) from the first sample
        keys = list(self.data[0].keys())

        # Create a list to hold NumPy arrays for each column
        numpy_arrays = []

        for key in keys:
            column_array = np.array([sample[key] for sample in self.data])
            numpy_arrays.append(column_array)

        return numpy_arrays

    def save_to_file(self, filename):
        # Save dataset to a file using pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.data, f)
            pickle.dump(self.encoders, f)

    @staticmethod
    def load_from_file(filename):
        # Load dataset from a file using pickle
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            encoders = pickle.load(f)
        dataset = MLDataSet()
        dataset.data = data
        dataset.encoders = encoders
        return dataset

    def save_to_json_file(self, filename):
        # Convert dataset to JSON-compatible format
        json_data = []
        for record in self.data:
            json_record = {}
            for key, value in record.items():
                if isinstance(value, bytes):
                    # Convert binary data to base64 encoded string
                    json_record[key] = base64.b64encode(value).decode('utf-8')
                elif isinstance(value, np.ndarray):
                    # Convert ndarray to list
                    json_record[key] = value.tolist()
                elif isinstance(value, np.int64):
                    # Convert np.int64 to Python int
                    json_record[key] = int(value)
                else:
                    # Other data types can be serialized directly
                    json_record[key] = value
            json_data.append(json_record)

        # Save to JSON file
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=4)

    @staticmethod
    def load_from_json_file(filename):
        # Load JSON data from file
        with open(filename, 'r') as f:
            json_data = json.load(f)

        # Convert JSON data back to original format
        data = []
        for json_record in json_data:
            record = {}
            for key, value in json_record.items():
                if isinstance(value, str) and value.startswith('base64:'):
                    # Decode base64 encoded string back to binary data
                    record[key] = base64.b64decode(value[7:])
                elif isinstance(value, list):
                    # Convert lists back to numpy arrays
                    record[key] = np.array(value)
                else:
                    # Other data types are loaded directly
                    record[key] = value
            data.append(record)

        # Create a new MLDataSet with the loaded data
        dataset = MLDataSet()
        dataset.data = data
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
