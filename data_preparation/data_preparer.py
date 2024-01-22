from mci_lib_ml.data_balancer import DataBalancer


class DataPreparer:

    @staticmethod
    def balance(dataset, filter_columns, balance_columns):
        return DataBalancer.balance(dataset, filter_columns, balance_columns)

    @staticmethod
    def fix_data_shape_by_truncation(dataset):
        # Find the minimum shape for each column in the dataset
        min_shapes = {}
        for column in dataset.data[0].keys():  # Assuming all samples have the same keys
            shapes = [sample[column].shape for sample in dataset.data]
            min_shapes[column] = tuple(min(s[i] for s in shapes) for i in range(len(shapes[0])))

        # Truncate data in each column to the minimum shape
        for sample in dataset.data:
            for column, min_shape in min_shapes.items():
                sample[column] = DataPreparer.truncate_to_shape(sample[column], min_shape)

    @staticmethod
    def truncate_to_shape(data, min_shape):
        # Truncate data to the specified minimum shape
        slices = tuple(slice(0, min_length) for min_length in min_shape)
        return data[slices]

    def generate_input_shapes(dataset):
        input_shapes = {}

        # Assuming all samples have the same set of keys
        if len(dataset.data) == 0:
            raise ValueError("Dataset is empty.")

        sample_keys = dataset.data[0].keys()

        # Determine the shape of each column
        for key in sample_keys:
            column_shapes = [sample[key].shape for sample in dataset.data]
            max_shape = max(column_shapes, key=lambda x: len(x))  # Choose the shape with the most dimensions
            input_shapes[key] = max_shape

        # Print shape info to the console
        for feature, shape in input_shapes.items():
            print(f"Shape for '{feature}' feature: {shape}")

        return input_shapes
