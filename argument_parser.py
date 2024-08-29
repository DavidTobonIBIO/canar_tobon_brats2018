import argparse


parser = argparse.ArgumentParser(description="PyTorch SegFormer3D Brats2018")

parser.add_argument(
    "--run-name",
    type=str,
    default="base_segformer",
    help="Name of the run for logging purposes",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=2,
    metavar="N",
    help="input batch size for training (default: 8)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=15,
    metavar="N",
    help="number of epochs to train (default: 15)",
)
parser.add_argument(
    "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.5,
    metavar="M",
    help="SGD momentum (default: 0.5)",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=2,
    metavar="M",
    help="learning rate decay factor (default: 2)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before " "logging training status",
)
parser.add_argument(
    "--model",
    type=str,
    default="segformer_pretrained",
    help="model to use (default: segformer)",
)
parser.add_argument(
    "--save", type=str, default="model.pt", help="file on which to save model weights"
)
parser.add_argument(
    "--wandb", action="store_true", default=False, help="use wandb for logging"
)
parser.add_argument(
    "--file",
    type=str,
    default="segformer3D_baseline.pt",
    help="file to use for prediction, remember to put .pt at the end.",
)

parser.add_argument(
    "--dataset", type=str, default="brats2018_seg", help="Select the Brats dataset"
)

parser.add_argument(
    "--loss", type=str, default="dice", help="Select the loss function"
)

parser.add_argument(
    '--rank', type=int, default=0,help='rank of the process'
)

