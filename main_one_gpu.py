import os
import yaml
import wandb
import torch
import torch.optim as optim
from utils.losses import build_loss_fn
from build_dataset_one_gpu import build_dataset, build_dataloader
from model import build_segformer3d_model, SegFormer3D
from argument_parser import parser
from utils.metrics import build_metric_fn
from trainer_one_gpu import Brats2018Trainer

args = parser.parse_args()


def main(epochs, log_interval):

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cpu")
    if args.cuda:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    # Wandb initialization
    if args.wandb:
        wandb.init(project="3d-segmentation", name=args.run_name)
        wandb.config.update(args)

    # Set random seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Define data paths
    file_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(file_path, "preproc_data")
    train_path = os.path.join(data_path, "train_data")
    valid_path = os.path.join(data_path, "val_data")

    # Define path for saving model
    os.makedirs("models", exist_ok=True)
    args.save = os.path.join(file_path, "models", args.save)

    # Define dataset and dataloader
    train_dataset = build_dataset(
        dataset_type=args.dataset, root_dir=train_path, is_train=True
    )

    valid_dataset = build_dataset(
        dataset_type=args.dataset, root_dir=valid_path, is_train=False
    )

    train_loader = build_dataloader(train_dataset)
    valid_loader = build_dataloader(valid_dataset)

    # Load the configuration file
    config_path = os.path.join(file_path, "configs", "config.yaml")
    config_dict = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    if args.pretrained == "":
        # Define the model
        model = build_segformer3d_model(config=config_dict)
    else:
        if args.pretrained == "best_segformer3d_brats_performance.pt":
            state_dict = torch.load(os.path.join(file_path, "models", args.pretrained))
            model = SegFormer3D()
            model.load_state_dict(state_dict)
            model.to(device)
        else:
            model_info_dict = torch.load(
                os.path.join(file_path, "models", args.pretrained)
            )
            state_dict = model_info_dict["model_state_dict"]
            model = SegFormer3D()
            model.load_state_dict(state_dict)
            model.to(device)
        print("Weights loaded successfully from", args.pretrained)
    # Model parameters
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = build_loss_fn(args.loss)
    training_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Define metrics

    metrics_dict = {
        "roi": [128, 128, 128],
        "sw_batch_size": 2,
    }

    metrics_fn = build_metric_fn(
        metric_type="sliding_window_inference", metric_arg=metrics_dict
    )

    trainer = Brats2018Trainer(
        config=config_dict,
        num_epochs=epochs,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        valid_loader=valid_loader,
        compute_metrics=True,
        metrics_fn=metrics_fn,
        training_scheduler=training_scheduler,
        log_interval=log_interval,
        device=device,
        save_path=args.save,
        wandb=args.wandb,
    )

    trainer.train()


if __name__ == "__main__":
    main(args.epochs, args.log_interval)
