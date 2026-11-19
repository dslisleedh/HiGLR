"""Five-task identity-inclusive augmentation with preserved rain axis."""
import random
from torch.utils.data import Dataset
from .r2r_rain_axis_augmentation import transform
class TaskAugmentationDataset5D(Dataset):
    def __init__(self, original): self.original=original
    def __len__(self): return len(self.original)
    def __getitem__(self,index):
        from utils import dataset_utils
        raw=self.original.sample_ids[index]['de_type']
        if raw not in range(7): raise ValueError(raw)
        previous=dataset_utils.random_augmentation
        def augmentation(*images):
            mode=random.choice((0,1,4,5) if raw==3 else tuple(range(8)))
            return [transform(im,mode) for im in images]
        dataset_utils.random_augmentation=augmentation
        try:return self.original[index]
        finally:dataset_utils.random_augmentation=previous
