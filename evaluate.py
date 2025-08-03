import argparse
import torch
from model import scoring_head
from dataset.dataset_fs1000 import FeatureDataset, av_collate_fn
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import spearmanr
import pandas as pd
import os

def extract_features_from_video(video_path, device):
    """
    TODO: implement feature extraction for arbitrary video.
    """
    raise NotImplementedError("Feature extraction not implemented")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum sample index to score (inclusive) from validation set (default: all samples)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./fs1000_result/checkpoint_pe.pth',
        help='Path to the model checkpoint'
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        required=True,
        help='Root directory of the FS800 dataset'
    )
    parser.add_argument(
        '--score_item_index',
        type=int,
        default=0,
        help='Index of the score dimension to evaluate (0-based)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Directory to save the output Excel file'
    )
    args = parser.parse_args()

    # Set device
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load validation dataset
    dataset = FeatureDataset(root_path=args.dataset_root, is_train=False)
    val_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=av_collate_fn
    )

    # Build and load the trained model
    model = scoring_head(depth=2, input_dim=768, dim=512, input_len=16, num_scores=1, bidirection=True)
    model.load_state_dict(torch.load(args.checkpoint, map_location=dev))
    model.to(dev).eval()

    # Lists to collect predictions and ground truths for overall metrics
    y_preds = []
    y_trues = []
    # List to collect per-sample records
    records = []

    # Evaluate samples from index 0 to limit
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if args.limit is not None and idx > args.limit:
                break
            audio_feat, video_feat, inv_audio_feat, inv_video_feat, audio_len, video_len, score, data_idx = batch
            audio_feat = audio_feat.to(dev)
            video_feat = video_feat.to(dev)
            inv_audio_feat = inv_audio_feat.to(dev)
            inv_video_feat = inv_video_feat.to(dev)
            # Perform inference
            pred_score = model(audio_feat, video_feat, audio_len)
            # Extract ground truth for the specified dimension
            if isinstance(score, torch.Tensor):
                gt = score[0, args.score_item_index].item()
            else:
                gt = score[args.score_item_index]
            # Store for overall metric computation
            y_preds.append(float(pred_score.item()))
            y_trues.append(float(gt))
            # Print numeric predictions and ground truth with fixed formatting
            print(f'Index {idx}: Predicted = {float(pred_score.item()):.4f}, Ground truth = {float(gt):.4f}')
            # Record this sample's result
            records.append({
                'Index': idx,
                'Predicted': float(pred_score.item()),
                'GroundTruth': float(gt)
            })

    # Convert collected lists to NumPy arrays for metric computation
    preds_np = np.array(y_preds)
    trues_np = np.array(y_trues)

    # Compute numeric metrics
    mse_val = float(np.mean((preds_np - trues_np) ** 2))
    mae_val = float(np.mean(np.abs(preds_np - trues_np)))

    # Compute Spearman correlation, handle constant arrays
    try:
        spearman_corr, _ = spearmanr(trues_np, preds_np)
        spearman_corr = np.array(spearman_corr).item()
    except Exception:
        spearman_corr = float('nan')

    # Prepare summary DataFrame
    summary_df = pd.DataFrame([{'MSE': mse_val, 'MAE': mae_val, 'Spearman': spearman_corr}])
    # Prepare per-sample DataFrame
    samples_df = pd.DataFrame(records)

    # Write to Excel with two sheets
    os.makedirs(args.output_dir, exist_ok=True)
    excel_path = os.path.join(args.output_dir, f"score_item_index_{args.score_item_index}.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        samples_df.to_excel(writer, sheet_name='PerSample', index=False)
    print(f"Metrics and per-sample results saved to {excel_path}")
    print(f"Overall metrics - MSE: {mse_val:.4f}, MAE: {mae_val:.4f}, Spearman: {spearman_corr:.4f}")

if __name__ == "__main__":
    main()
