import pandas as pd
from torch.utils.data import WeightedRandomSampler


def make_weighted_sampler(df, label_column='diagnosis'):
    labels = df[label_column].values
    class_sample_count = pd.Series(labels).value_counts().sort_index()
    weights = 1. / class_sample_count
    sample_weights = [weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler