import numpy as np
from torch.utils.data import Subset
from torch.utils.data import DataLoader


def create_subset_loader(train_loader, subset_ratio=0.25):
    training_dataset = train_loader.dataset
    subset_size = int(len(training_dataset) * subset_ratio)
    indices = np.random.choice(len(training_dataset), size=subset_size, replace=False)
    subset_train_dataset = Subset(training_dataset, indices)
    subset_train_loader = DataLoader(subset_train_dataset, batch_size=train_loader.batch_size, shuffle=True)

    return subset_train_loader
