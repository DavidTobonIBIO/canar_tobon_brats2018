import torch
from torch.utils.data.distributed import DistributedSampler

from monai.data import DataLoader

from utils.augmentations import build_augmentations
from argument_parser import parser

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {"num_workers": 8, "pin_memory": True} if args.cuda else {}


######################################################################
def build_dataset(dataset_type: str, root_dir: str, is_train: bool):
    if dataset_type == "brats2018_seg":
        from BraTs2018_dataset import Brats2018Dataset

        dataset = Brats2018Dataset(
            root_dir=root_dir,
            is_train=is_train,
            transform=build_augmentations(train=is_train),
        )
        return dataset
    else:
        raise ValueError("only brats2018 dataset is supported.")


######################################################################
def build_dataloader(dataset) -> DataLoader:
    """Builds the dataloader for the given dataset.

    Args:
        dataset (Dataset): The dataset for which the dataloader is being built.

    Returns:
        DataLoader: The dataloader for the specified dataset.
    """
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=DistributedSampler(dataset) if args.cuda else None,
        **kwargs  # kwargs are applied directly for CUDA-related settings
    )
    return dataloader
