import os
from tqdm import tqdm
import torch
from utils.losses import build_loss_fn
from build_dataset_one_gpu import build_dataset, build_dataloader
from model import SegFormer3D
from argument_parser import parser
from utils.metrics import build_metric_fn
from trainer_lindo import Brats2018Trainer
from utils.visualize_data import (
    animate,
    viz_animation,
    get_max_slice,
    viz_figure,
    viz_figure_all_labels,
    load_label,
    load_volume,
)

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cpu")
if args.cuda:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Set random seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

file_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(file_path, "preproc_data")
train_path = os.path.join(data_path, "train_data")
valid_path = os.path.join(data_path, "val_data")

models_path = os.path.join(file_path, "models")
model_path = os.path.join(models_path, args.save)

valid_dataset = build_dataset(
    dataset_type=args.dataset, root_dir=valid_path, is_train=False
)

valid_loader = build_dataloader(valid_dataset)
model_info_dict = torch.load(model_path)
state_dict = model_info_dict["model_state_dict"]
model = SegFormer3D()
model.load_state_dict(state_dict)
model.to(device)

metrics_dict = {
    "roi": [128, 128, 128],
    "sw_batch_size": args.batch_size,
}

metrics_fn = build_metric_fn(
    metric_type="sliding_window_inference", metric_arg=metrics_dict
)

model.eval()

avg_dice = 0.0

with torch.no_grad():
    with tqdm(total=len(valid_loader), desc="Validation", unit="batch") as pbar:
        for idx, raw_data in enumerate(valid_loader):
            data = raw_data["image"].to(device)
            labels = raw_data["label"].to(device)

            preds = model(data)

            avg_dice += metrics_fn(data, labels, model=model)

            pbar.set_postfix(
                {
                    "val_dice": avg_dice / (idx + 1),
                }
            )
            pbar.update(1)

avg_dice /= len(valid_loader)
        
print(f"Average Dice: {avg_dice}")
print(args.batch_size)
