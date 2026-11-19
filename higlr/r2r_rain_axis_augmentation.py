"""AIO identity-inclusive augmentation, preserving the rain streak axis."""
import random
import numpy as np
from torch.utils.data import Dataset

VARIANT = 'r2r_naf_identity_rain_axis'
POLICY = {
    'model_variant': VARIANT,
    'fusion_kernel': 1,
    'augmentation_policy': 'identity_all_tasks_rain_axis_v1',
    'augmentation_modes': {'noise': list(range(8)), 'rain': [0, 1, 4, 5], 'haze': list(range(8))},
    'augmentation_sampling': 'uniform within each task; same transform for input and target',
}


def transform(image, mode):
    image = np.asarray(image)
    if mode not in range(8):
        raise ValueError(mode)
    result = np.rot90(image, k=mode // 2, axes=(0, 1))
    if mode % 2:
        result = np.flipud(result)
    return result.copy()


def augment_pair(*images, de_type):
    if de_type not in range(5):
        raise ValueError(de_type)
    modes = (0, 1, 4, 5) if de_type == 3 else tuple(range(8))
    mode = random.choice(modes)
    return [transform(image, mode) for image in images]


class TaskAugmentationDataset(Dataset):
    """Delegate crop/noise/loading unchanged; replace only the crop transform.

    Each DataLoader worker calls __getitem__ synchronously. The temporary module
    callback is restored even on failure and never modifies upstream files.
    """
    def __init__(self, original):
        self.original = original

    def __len__(self):
        return len(self.original)

    def __getitem__(self, index):
        from utils import dataset_utils
        de_type = self.original.sample_ids[index]['de_type']
        previous = dataset_utils.random_augmentation
        def augmentation(*images):
            return augment_pair(*images, de_type=de_type)
        dataset_utils.random_augmentation = augmentation
        try:
            return self.original[index]
        finally:
            dataset_utils.random_augmentation = previous
