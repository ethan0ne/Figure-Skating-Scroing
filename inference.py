import argparse
import torch
from model import scoring_head
from dataset.dataset_fs800 import FeatureDataset, av_collate_fn
from torch.utils.data import DataLoader

def extract_features_from_video(video_path, device):
    """
    TODO: implement feature extraction for arbitrary video.
    """
    raise NotImplementedError("Feature extraction not implemented")

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--infer_video_index',
        type=int,
        help='Index of the video in the validation dataset to score'
    )
    group.add_argument(
        '--video_path',
        type=str,
        help='Path to a video file to score (must implement extract_features_for_video)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./fs800_result/checkpoint_pe.pth',
        help='Path to the model checkpoint'
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        help='Root directory of the FS800 dataset (required if using --infer_video_index)',
        required=False
    )
    args = parser.parse_args()

    # Set device
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare features either from dataset or arbitrary video
    if args.video_path:
        # Use the stub to extract features from a new video file
        audio_feat, video_feat, inv_audio_feat, inv_video_feat, audio_len, video_len = extract_features_from_video(args.video_path, dev)
        audio_feat = audio_feat.unsqueeze(0).to(dev)
        video_feat = video_feat.unsqueeze(0).to(dev)
        inv_audio_feat = inv_audio_feat.unsqueeze(0).to(dev)
        inv_video_feat = inv_video_feat.unsqueeze(0).to(dev)
        audio_len = [audio_len]
        video_len = [video_len]
    else:
        # Load validation dataset and fetch by index
        if not args.dataset_root:
            parser.error("--dataset_root is required when using --infer_video_index")
        val_dataset = FeatureDataset(root_path=args.dataset_root, is_train=False)
        # Use DataLoader to batch and collate features correctly
        val_loader = DataLoader(val_dataset, batch_size=1, num_workers=0, collate_fn=av_collate_fn)
        for idx, batch in enumerate(val_loader):
            if idx == args.infer_video_index:
                audio_feat, video_feat, inv_audio_feat, inv_video_feat, audio_len, video_len, score, data_idx = batch
                break
        audio_feat = audio_feat.to(dev)
        video_feat = video_feat.to(dev)
        inv_audio_feat = inv_audio_feat.to(dev)
        inv_video_feat = inv_video_feat.to(dev)
        # audio_len and video_len are already lists from collate function

    # Build and load the trained model
    model = scoring_head(depth=2, input_dim=768, dim=512, input_len=16, num_scores=1, bidirection=True)
    model.load_state_dict(torch.load(args.checkpoint, map_location=dev))
    model.to(dev).eval()

    # Perform inference
    with torch.no_grad():
        pred_score = model(audio_feat, video_feat, audio_len)

    print(f'Predicted score for video index {args.infer_video_index if args.infer_video_index is not None else "from video_path"}: {pred_score.item()}')
    if args.infer_video_index is not None:
        # Show ground truth score for debugging (handle tensor or list)
        if hasattr(score, 'item'):
            gt = score.item()
        elif isinstance(score, (list, tuple)):
            if len(score) == 1:
                first = score[0]
                gt = first.item() if hasattr(first, 'item') else first
            else:
                gt = [s.item() if hasattr(s, 'item') else s for s in score]
        else:
            gt = score
        print(f'Expected score for video index {args.infer_video_index}: {gt}')

if __name__ == "__main__":
    main()
