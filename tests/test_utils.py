import unittest
import numpy as np
from utils import split_data_hetero
"""
class TestHeteroSplit(unittest.TestCase):
    def setUp(self):
        self.data = []
        for label in range(5):
            for i in range(20):
                self.data.append((f"img_{label}_{i}", label))
    
    def test_all_samples_assigned(self):
        groups = split_data_hetero(self.data, num_groups = 5)
        assigned = []
        for group in groups:
            assigned.extend(group)
        
        self.assertEqual(len(assigned), len(self.data))
        self.assertEqual(set(assigned), set(self.data))
    
    def test_correct_number_of_groups(self):
        groups = split_data_hetero(self.data, num_groups = 7)
        self.assertEqual(len(groups), 7)
    
    def test_minimum_samples(self):
        groups = split_data_hetero(self.data, num_groups = 5, min_samples = 10)
        for group in groups:
            self.assertGreaterEqual(len(group), 10)"""