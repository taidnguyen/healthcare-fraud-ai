import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FraudDataset(Dataset):
    """Import CMS and LEIE combined dataset."""

    def __init__(self, csv_file, split, split_percent, batch_size, batchseed=0):
        """
        Instantiate FraudDataset object.
        :param csv_file: location for combined CMS, LEIE dataset
        :param split: designation of which dataset to use: {'train', 'val'}
        :param split_percent: percentage of data to use as training vs. validation (float in (0, 1))
        :param batch_size: an integer to show the amount of data to train on 
        :param seed: random seed for shuffling data (default: 0)
        """
        if split not in ['val', 'train']:
            raise Exception(f'Invalid dataset split: {split}. Provide \'train\' or \'val\' as parameter value.')

        self.split = split

        if not 0 < split_percent < 1: 
            raise Exception(f'Invalid split fraction: {split_percent}. Provide value in range (0, 1).')
        
        self.seed = batchseed
        self.split_percent = split_percent
        
        df = pd.read_csv(csv_file).sample(1, random_state=np.random.RandomState(self.seed))  # Load & shuffle data
        df_len = len(df.index)
        
        self.split_idx = int(self.split_percent * df_len)
        self.df = df[:self.split_idx] if self.split == 'train' else df[self.split_idx:]

        if not 0 <= batch_size < df_len:
            raise Exception(f'Batch size of {batch_size} is invalid. Must be between range [0, {df_len})')
        
        self.bsz = batch_size


    def __len__(self):
        return len(self.df.index)


    def __getitem__(self, idx):
        """
        Implements __getitem__ method of Dataset class
        :param idx: starting index of next sample
        :param bsz: batch size
        :return: sample - dictionary of 'predictors' (tensor) and 'label' (tensor)
        """
        predictors = torch.tensor(self.df.iloc[idx:idx+self.bsz, self.df.columns != 'label'].values())
        label = torch.tensor(self.df.iloc[idx:idx+self.bsz, self.df.columns == 'label'].values())
        sample = {'predictors': predictors, 'label': label}
        
        return sample
