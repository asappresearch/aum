from torch.utils.data import Dataset


class DatasetWithIndex(Dataset):
    """
    A thin wrapper over a pytorch dataset that includes the sample index as the last element
    of the tuple returned.
    """
    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        return (*self.base_dataset[index], index)
