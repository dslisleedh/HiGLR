"""R2R AIO lists/loaders plus the established HiGLR data transforms."""
from pathlib import Path
import sys
from types import SimpleNamespace

from .aio5_task_augmentation import TaskAugmentationDataset5D
from .cdd11_data import CDD11Dataset
from .decoder_sas_aio5_data import prepare_finetune


def build_dataset(name, root, patch_size, lists=None):
    root = Path(root).resolve(strict=True)
    if name == 'cdd11':
        return CDD11Dataset(root, patch_size)
    if name not in ('aio3', 'aio5'):
        raise ValueError('Supported datasets: aio3, aio5, cdd11')
    upstream = Path(__file__).resolve().parents[1] / 'reference' / 'r2r'
    sys.path.insert(0, str(upstream))
    from utils.dataset_utils import PromptTrainDataset, PromptTrainDataset5D
    tasks = ['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze']
    if name == 'aio5':
        tasks += ['deblur', 'lowlight']
    args = SimpleNamespace(
        patch_size=patch_size, de_type=tasks,
        data_file_dir=str(Path(lists).resolve() if lists else upstream / 'data_dir') + '/',
        **{key+'_dir': str(root / directory) + '/' for key, directory in
           [('denoise', 'Denoise'), ('derain', 'Derain'), ('dehaze', 'Dehaze'),
            ('deblur', 'Deblur'), ('lowlight', 'Lowlight')]})
    dataset = (PromptTrainDataset if name == 'aio3' else PromptTrainDataset5D)(args)
    # Canonicalize the list order: upstream os.listdir/random.shuffle order is
    # otherwise host dependent. The distributed sampler performs epoch shuffling.
    dataset.sample_ids.sort(key=lambda item: (item['de_type'], item['clean_id']))
    if patch_size == 256:
        dataset = prepare_finetune(dataset)
    return TaskAugmentationDataset5D(dataset)


def class_ids(raw_ids, dataset):
    return raw_ids if dataset == 'cdd11' else raw_ids.where(raw_ids >= 3, 2) - 2
