import unittest

from mci_lib_ml.data_balancer import DataBalancer


class TestDataPreparer(unittest.TestCase):

    def test_balance(self):
        # Create a mock dataset
        dataset = [
            {'file_path': 'file1.wav', 'labels': ['actor-1', 'small-size', 'light-weight']},
            {'file_path': 'file2.wav', 'labels': ['actor-2', 'small-size', 'light-weight']},
            {'file_path': 'file3.wav', 'labels': ['actor-2', 'small-size', 'light-weight']},
            {'file_path': 'file4.wav', 'labels': ['actor-3', 'small-size', 'light-weight']},
            {'file_path': 'file5.wav', 'labels': ['actor-3', 'small-size', 'light-weight']},
            {'file_path': 'file6.wav', 'labels': ['actor-3', 'small-size', 'light-weight']},
            {'file_path': 'file7.wav', 'labels': ['actor-4', 'small-size', 'heavy-weight']},
            {'file_path': 'file8.wav', 'labels': ['actor-5', 'large-size', 'light-weight']},
            # Additional records...
        ]

        # Apply balance function
        filter_labels = ['small-size', 'light-weight']
        balance_labels = ['actor-1', 'actor-2', 'actor-3']
        balanced_dataset = DataBalancer.balance(dataset, filter_labels, balance_labels)

        # Expected filenames in the balanced dataset
        expected_actors = {'actor-1', 'actor-2', 'actor-3'}

        # Extract labels of actors from the balanced dataset
        actors_in_balanced_dataset = set()
        for record in balanced_dataset:
            if 'small-size' in record['labels'] and 'light-weight' in record['labels']:
                actors = [label for label in record['labels'] if label.startswith('actor')]
                actors_in_balanced_dataset.update(actors)

        # Assert that the balanced dataset contains the expected filenames
        self.assertEqual(expected_actors, actors_in_balanced_dataset)

        # Assert that the size of the balanced dataset is as expected
        expected_size = len(expected_actors)
        self.assertEqual(len(balanced_dataset), expected_size)


if __name__ == '__main__':
    unittest.main()
