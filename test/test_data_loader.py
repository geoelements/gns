import unittest
import numpy as np
import os
import tempfile
import shutil
import sys

# Add parent directory to Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from gns.data_loader import SamplesDataset, TrajectoriesDataset

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        # Create a dummy dataset
        self.dummy_data = [(np.random.rand(10, 3, 2), i % 3) for i in range(5)]
        self.data_path = os.path.join(self.temp_dir, 'data.npz')
        np.savez(self.data_path, *self.dummy_data)

        # Set input_length_sequence
        self.input_length_sequence = 5

    def tearDown(self):
        # Delete temporary directory
        shutil.rmtree(self.temp_dir)

    def test_samples_dataset(self):
        dataset = SamplesDataset(self.data_path, self.input_length_sequence)
        self.assertEqual(len(dataset), 25)  # 5 (trajectories) * (10 (positions) - 5 (input_length_sequence))
        for i in range(len(dataset)):
            ((positions, particle_type, n_particles_per_example), label) = dataset[i]
            self.assertEqual(positions.shape, (3, self.input_length_sequence, 2))  # Check positions shape
            self.assertEqual(particle_type.shape, (3,))  # Check particle_type shape
            self.assertEqual(n_particles_per_example, 3)  # Check number of particles per example
            self.assertEqual(label.shape, (3, 2))  # Check label shape

    def test_trajectories_dataset(self):
        dataset = TrajectoriesDataset(self.data_path)
        self.assertEqual(len(dataset), 5)  # We have 5 trajectories
        for i in range(len(dataset)):
            (positions, particle_type, n_particles_per_example) = dataset[i]
            self.assertEqual(positions.shape, (3, 10, 2))  # Check positions shape
            self.assertEqual(particle_type.shape, (3,))  # Check particle_type shape
            self.assertEqual(n_particles_per_example, 3)  # Check number of particles per example

if __name__ == '__main__':
    unittest.main()
