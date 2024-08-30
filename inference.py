import os
import torch
from utils.losses import build_loss_fn
from build_dataset import build_dataset, build_dataloader
from model import build_segformer3d_model
from argument_parser import parser
from utils.metrics import build_metric_fn
from trainer_lindo import Brats2018Trainer
from utils.visualize_data import animate, viz_animation, get_max_slice, viz_figure, viz_figure_all_labels, load_label, load_volume

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cpu")
if args.cuda:
    device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    
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

model = torch.load(model_path)

metrics_dict = {
        "roi": [128, 128, 128],
        "sw_batch_size": 2,
    }

metrics_fn = build_metric_fn(
    metric_type="sliding_window_inference", metric_arg=metrics_dict
)

model.eval()