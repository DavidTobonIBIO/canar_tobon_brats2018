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


fig, axis = plt.subplots(args.batch_size, 3, figsize=(12, 9))

axis[0, 0].set_title("MRI Slice")
axis[0, 1].set_title("Ground Truth")
axis[0, 2].set_title("Prediction")

tc_label_idx = 0
wt_label_idx = 1
et_label_idx = 2
flair_vol_idx = 0

for i in range(args.batch_size):
    label = labels[i, ...]
    pred = preds[i, ...]
    max_slice_idx = get_max_slice(label[wt_label_idx])
    
    axis[i, 0].imshow(data[i, flair_vol_idx, max_slice_idx], cmap="gray")
    
    labeled_img = np.copy(data[i, flair_vol_idx, max_slice_idx])
    labeled_img = np.repeat(labeled_img[:, :, np.newaxis], 3, axis=2)

    colors = [(0, 255, 255), (255, 255, 0), (255, 0, 0)]
    label_order = [1, 0, 2]
    for i in label_order:
        color = colors[i]
        l = label[i, ...]
        labeled_img[l[max_slice_idx] == True] = color
    
    axis[i, 1].imshow(labeled_img)
    
    preds_img = np.copy(data[i, flair_vol_idx, max_slice_idx])
    preds_img = np.repeat(preds_img[:, :, np.newaxis], 3, axis=2)

    for i in label_order:
        color = colors[i]
        l = pred[i, ...]
        preds_img[l[max_slice_idx] == True] = color
        
    axis[i, 2].imshow(preds_img)
    
    axis[i, 0].axis("off")
    axis[i, 1].axis("off")
    axis[i, 2].axis("off")

plt.tight_layout()
plt.show()
plt.savefig(os.path.join(visualizations_dir, "pred.png"))
