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
device = torch.device("cuda" if args.cuda else "cpu")

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

def calculate_batch_mean_pixel_accuracy(preds, labels):
    """
    Calculate the mean pixel accuracy for a batch of data.
    
    Args:
    - preds (numpy array): Predictions with shape (batch_size, num_classes, height, width).
    - labels (numpy array): Ground truth labels with shape (batch_size, num_classes, height, width).
    
    Returns:
    - mean_pixel_accuracy (float): Mean pixel accuracy.
    """
    correct_pixels = 0
    total_pixels = 0
    
    batch_size = preds.shape[0]
    
    for i in range(batch_size):
        pred = np.argmax(preds[i], axis=0)
        label = np.argmax(labels[i], axis=0)
        
        correct_pixels += np.sum(pred == label)
        total_pixels += np.size(label)
    
    mean_pixel_accuracy = correct_pixels / total_pixels
    return mean_pixel_accuracy

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

def calculate_mean_pixel_accuracy():
    total_accuracy = 0.0

    with torch.no_grad():
        with tqdm(total=len(valid_loader), desc="Validation", unit="batch") as pbar:
            for idx, raw_data in enumerate(valid_loader):
                data = raw_data["image"].to(device)
                labels = raw_data["label"].to(device)

                preds = model(data)
                preds = torch.softmax(preds, dim=1).cpu().numpy()
                labels = labels.cpu().numpy()
                
                accuracy = calculate_batch_mean_pixel_accuracy(preds, labels)
                total_accuracy += accuracy

                pbar.set_postfix(
                    {
                        "val_pixel_accuracy": total_accuracy / (idx + 1),
                    }
                )
                pbar.update(1)

    total_accuracy /= len(valid_loader)

    print(f"Mean Pixel Accuracy: {total_accuracy}")

if args.compute_metrics:
    calculate_avg_dice()
    calculate_mean_pixel_accuracy()

raw_data = next(iter(valid_loader))
data = raw_data["image"].to(device)
preds = model(data)

# visualize the data
visualizations_dir = os.path.join(file_path, "visualizations", "preds")
os.makedirs(visualizations_dir, exist_ok=True)

data = raw_data["image"].cpu().numpy()
labels = raw_data["label"].cpu().numpy()
preds = torch.softmax(preds, dim=1) >= 1
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
    for j in label_order:
        color = colors[j]
        l = label[j, ...]
        labeled_img[l[max_slice_idx] == True] = color
    
    axis[i, 1].imshow(labeled_img)
    
    preds_img = np.copy(data[i, flair_vol_idx, max_slice_idx])
    preds_img = np.repeat(preds_img[:, :, np.newaxis], 3, axis=2)

    for j in label_order:
        color = colors[j]
        l = pred[j, ...]
        preds_img[l[max_slice_idx] == True] = color
        
    axis[i, 2].imshow(preds_img)
    
    axis[i, 0].axis("off")
    axis[i, 1].axis("off")
    axis[i, 2].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(visualizations_dir, f"pred_{args.save}.png"))
plt.show()
