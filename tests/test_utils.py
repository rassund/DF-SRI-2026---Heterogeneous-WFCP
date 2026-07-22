import unittest
import numpy as np
from utils import split_data_hetero
from collections import Counter

class TestHeteroSplit(unittest.TestCase):
    def setUp(self):
        self.data = []
        for label in range(10):
            for i in range(500):
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
            self.assertGreaterEqual(len(group), 10)
    
    def test_class_counts_preserved(self):
        original = Counter(label for _, label in self.data)
        groups = split_data_hetero(self.data, num_groups = 5)
        new = Counter()

        for group in groups:
            new.update(label for _, label in group)

        self.assertEqual(original, new)
    
    def test_small_alpha_is_non_iid(self):
        groups = split_data_hetero(self.data, num_groups = 5, alpha = 0.05)

        global_dist = Counter(label for _, label in self.data)

        for label in global_dist:
            global_dist[label] /= len(self.data)

        heterogeneous = False

        for group in groups:
            counts = Counter(label for _, label in group)

            for label in global_dist:
                local = counts[label] / len(group)

                if abs(local - global_dist[label]) > 0.2:
                    heterogeneous = True

        self.assertTrue(heterogeneous)
    
    def test_large_alpha_is_nearly_iid(self):
        groups = split_data_hetero(self.data, num_groups = 5, alpha = 100)
        for group in groups:
            labels = {label for _, label in group}
            self.assertGreaterEqual(len(labels), 4)
    
    def test_group_sizes_differ(self):
        groups = split_data_hetero(self.data, num_groups = 5, alpha = 0.5)
        sizes = [len(group) for group in groups]
        self.assertGreater(max(sizes), min(sizes))
    
    def test_no_empty_groups(self):
        groups = split_data_hetero(self.data, num_groups = 5, min_samples = 5)
        for group in groups:
            self.assertGreater(len(group), 0)