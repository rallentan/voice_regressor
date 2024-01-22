import numpy as np


class Ops:

    @staticmethod
    def dict_to_lists(dictionary):
        lists = [list(item) for item in dictionary.values()]
        return lists

    @staticmethod
    def normalize_mean_and_variance(numpy_array):
        return (numpy_array - np.mean(numpy_array)) / np.std(numpy_array)
