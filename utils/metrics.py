import torch
import torch.nn as nn
from typing import Dict
from monai.metrics import DiceMetric
from monai.transforms import Compose
from monai.data import decollate_batch
from monai.transforms import Activations, AsDiscrete
from monai.inferers import sliding_window_inference

class SlidingWindowInference:
    def __init__(self, roi: tuple, sw_batch_size: int):
        self.dice_metric = DiceMetric(
            include_background=True, reduction="mean_batch", get_not_nans=False
        )
        self.post_transform = Compose(
            [
                Activations(sigmoid=True),
                AsDiscrete(argmax=False, threshold=0.5),
            ]
        )
        self.sw_batch_size = sw_batch_size
        self.roi = roi

    def __call__(self, val_inputs: torch.Tensor, val_labels: torch.Tensor, model: nn.Module):
        try:
            # Reset dice metric for the new batch
            print("Resetting Dice metric...")
            self.dice_metric.reset()

            # Perform sliding window inference
            print(f"Performing sliding window inference with ROI size: {self.roi} and SW batch size: {self.sw_batch_size}")
            logits = sliding_window_inference(
                inputs=val_inputs,
                roi_size=self.roi,
                sw_batch_size=self.sw_batch_size,
                predictor=model,
                overlap=0.5,
            )
            print("Sliding window inference completed.")

            # Debugging tensor shapes
            print(f"Logits shape: {logits.shape}")
            print(f"Val inputs shape: {val_inputs.shape}")
            print(f"Val labels shape: {val_labels.shape}")

            # Decollate the batch
            print("Decollating batch...")
            val_labels_list = decollate_batch(val_labels)
            val_outputs_list = decollate_batch(logits)

            # Debugging post-processed predictions
            print("Post-processing predictions...")
            val_output_convert = [
                self.post_transform(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            print(f"Post-processed predictions count: {len(val_output_convert)}")

            # Compute the Dice metric
            print("Computing Dice metric...")
            self.dice_metric(y_pred=val_output_convert, y=val_labels_list)
            acc = self.dice_metric.aggregate().cpu().numpy()

            print(f"Dice scores per channel: {acc}")
            avg_acc = acc.mean()
            print(f"Average Dice score: {avg_acc}")

            return avg_acc

        except Exception as e:
            print(f"An error occurred: {e}")
            raise

def build_metric_fn(metric_type: str, metric_arg: Dict = None):
    if metric_type == "sliding_window_inference":
        return SlidingWindowInference(
            roi=metric_arg["roi"],
            sw_batch_size=metric_arg["sw_batch_size"],
        )
    else:
        raise ValueError("must be cross sliding_window_inference!")
