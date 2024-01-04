import numpy as np
from sklearn.model_selection import train_test_split

from ml_data_set import MLDataSet


class DataPreparer:
    @staticmethod
    def balance(dataset, filter_columns, balance_columns):
        """
        Balance a subset of the dataset based on specified filter and balance columns,
        then return the entire dataset with the balanced subset included.
        """
        # Filter the dataset based on filter_columns
        filtered_dataset = DataPreparer.filter_by_columns(dataset, filter_columns)

        # Filter the dataset for remaining records not included in filtered_dataset
        remaining_dataset = DataPreparer.filter_out_columns(dataset, filter_columns)

        # Group the filtered dataset by balance_columns
        grouped_dataset = DataPreparer.group_by_columns(filtered_dataset, balance_columns)

        # Find the minimum size among the groups
        min_size = DataPreparer.min_length(grouped_dataset)

        # Balance the dataset by trimming each group to the minimum size
        balanced_subset = DataPreparer.flatten([records[:min_size] for records in grouped_dataset.values()])

        # Combine the balanced subset with the remaining dataset
        return MLDataSet.from_list(balanced_subset.data + remaining_dataset.data)


    # Private Methods #

    @staticmethod
    def filter_by_columns(dataset: MLDataSet, columns):
        return MLDataSet.from_list([record for record in dataset.data if all(record.get(col) for col in columns)])

    @staticmethod
    def filter_out_columns(dataset: MLDataSet, columns):
        # This function will filter out records that match the specified columns
        return MLDataSet.from_list([record for record in dataset.data if not all(record.get(col) for col in columns)])

    @staticmethod
    def group_by_columns(dataset: MLDataSet, columns):
        grouped = {}
        for record in dataset.data:
            # Create a key based on the combination of balance_columns values
            key = tuple(record.get(col) for col in columns)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(record)
        return grouped

    @staticmethod
    def min_length(groups):
        return min(len(group) for group in groups.values())

    @staticmethod
    def flatten(list_of_lists):
        return MLDataSet.from_list([item for sublist in list_of_lists for item in sublist])