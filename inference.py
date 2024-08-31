import os
from tqdm import tqdm
import torch
from build_dataset_one_gpu import build_dataset, build_dataloader
from model import SegFormer3D
from argument_parser import parser
from utils.metrics import build_metric_fn
from utils.visualize_data import get_max_slice
import numpy as np
import matplotlib.pyplot as plt

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
model = SegFormer3D()
if args.save == "best_segformer3d_brats_performance.pt":
    state_dict = torch.load(model_path)
else:
    model_info_dict = torch.load(model_path)
    state_dict = model_info_dict["model_state_dict"]

model.load_state_dict(state_dict)
model.to(device)
print("Weights loaded successfully from", args.save)
metrics_dict = {
    "roi": [128, 128, 128],
    "sw_batch_size": args.batch_size,
}

metrics_fn = build_metric_fn(
    metric_type="sliding_window_inference", metric_arg=metrics_dict
)

model.eval()

def calculate_avg_dice():
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

if args.compute_metrics: 
    calculate_avg_dice()

raw_data = next(iter(valid_loader))
data = raw_data["image"].to(device)
preds = model(data)

# visualize the data
visualizations_dir = os.path.join(file_path, "visualizations", "preds")
os.makedirs(visualizations_dir, exist_ok=True)

data = raw_data["image"].cpu().numpy()
labels = raw_data["label"].cpu().numpy()
preds = torch.softmax(preds, dim=1) > 0.5
preds = preds.detach().cpu().numpy()

print("Data shape: ", data.shape)
print("Labels shape: ", labels.shape)
print("Preds shape: ", preds.shape)

idx = get_max_slice(preds[0, 1])
label = labels[0, 1][idx]
pred = preds[0, 1][idx]

fig, axis = plt.subplots(1, 2, figsize=(10, 5))

axis[0].imshow(label, cmap="gray")
axis[1].imshow(pred, cmap="gray")
axis[0].axis("off")
axis[1].axis("off")

plt.tight_layout()

plt.savefig(os.path.join(visualizations_dir, "pred.png"))