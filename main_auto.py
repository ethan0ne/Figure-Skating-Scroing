import os
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from model import scoring_head
from dataset.dataset_fs800 import FeatureDataset, av_collate_fn

def pearson_loss(pred, target):
    vx = pred - pred.mean()
    vy = target - target.mean()
    corr = (vx * vy).sum() / (torch.sqrt((vx**2).sum()) * torch.sqrt((vy**2).sum()) + 1e-8)
    return 1 - corr

# Spearman loss using approximate differentiable ranks
def spearman_loss(pred, target, scale=1.0):
    # approximate ranks using pairwise sigmoid
    def _approx_rank(x):
        diff = x.unsqueeze(1) - x.unsqueeze(0)
        return torch.sigmoid(-scale * diff).sum(dim=1)
    pred_ranks = _approx_rank(pred)
    targ_ranks = _approx_rank(target)
    vx = pred_ranks - pred_ranks.mean()
    vy = targ_ranks - targ_ranks.mean()
    corr = (vx * vy).sum() / (torch.sqrt((vx**2).sum()) * torch.sqrt((vy**2).sum()) + 1e-8)
    return 1 - corr

def get_dataloaders(root, batch_size=16, num_workers=0):
    train_ds = FeatureDataset(root_path=root, is_train=True)
    val_ds = FeatureDataset(root_path=root, is_train=False)
    return (
        DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=av_collate_fn),
        DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, collate_fn=av_collate_fn),
    )

def train_one_epoch(model, dataloader, criterion, optimizer, sp_weight, score_index):
    model.train()
    total_loss = 0.0
    for audio, video, inv_audio, inv_video, audio_len, video_len, score, _ in dataloader:
        audio, video, inv_audio, inv_video = to_device(audio, video, inv_audio, inv_video)
        target = score[score_index].to(device)
        optimizer.zero_grad()
        output = model(audio, video, audio_len)
        loss = criterion(output, target) + sp_weight * spearman_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Set fixed seed for reproducibility
seed = 2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# detect MPS (Apple Silicon) or fallback to CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# utility to move tensors to the selected device
to_device = lambda *args: [x.to(device) for x in args]

def validation(dataloader, model, criterion, score_index):
    model.eval()
    val_loss = 0
    val_truth = []
    val_pred = []

    for audio_feature, video_feature, inv_audio_feature, inv_video_feature, audio_len, video_len, score, data_index in dataloader:
        batch_size = audio_feature.size(0)
        audio_feature, video_feature, inv_audio_feature, inv_video_feature = to_device(audio_feature, video_feature, inv_audio_feature, inv_video_feature)
        target = score[score_index].to(device)

        with torch.no_grad():
            output = model(audio_feature, video_feature, audio_len)
        val_pred.append(output.detach().cpu().numpy())
        val_truth.append(target.cpu().numpy())

        loss = criterion(output, target)
        val_loss += loss.item() * batch_size

    val_truth = np.concatenate(val_truth)
    val_pred = np.concatenate(val_pred)
    print("val_pred sample:", val_pred[:10])
    print("val_truth sample:", val_truth[:10])
    spear = spearmanr(val_truth, val_pred)
    dataset_size = len(dataloader.dataset)
    val_loss = val_loss / dataset_size
    return val_loss, spear.correlation

if __name__ == "__main__":
    FS1000_dataset_dir = "/Volumes/Data (Samsung PSSD T7 - 2 TB of Ethan)/Study/KCL/MSc Individual Project/References/FS1000 Dataset/"
    # training setup
    epochs = 500

    # select hyperparameters for current score
    learning_rates = [5e-5, 2e-5, 2e-5, 5e-5, 1e-4, 2e-5, 5e-5]
    weight_decays = [1e-5, 5e-6, 1e-5, 1e-6, 5e-6, 1e-5, 1e-6]
    spearman_weights = [1.0, 0.7, 0.5, 0.3, 0.1, 0.5, 0.1]
    #noise_stds = [0, 0.01, 0.05, 0.1, 0.2, 0.5]
    noise_stds = [0, 0.01, 0.05, 0.2, 0.5]
    score_index_limit = 7
    result = []

    no_improvements_tolerance = 60

    # build datasets and dataloaders
    train_dataloader, val_dataloader = get_dataloaders(FS1000_dataset_dir)

    for score_index in range(0, 2):
        for noise_std in noise_stds:
            # model
            model = scoring_head(depth=2, input_dim=768, dim=512, input_len=16, num_scores=1, bidirection=True,
                                 noise_std=noise_std).to(device)
            print("=" * 25)
            print(f"Model output: {model.output}, fusion: {getattr(model, 'fusion_linear', None)}")

            # select lr, weight decay, and Spearman weight for this score
            lr = learning_rates[score_index]
            wd = weight_decays[score_index]
            sp_weight = spearman_weights[score_index]

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
            criterion = torch.nn.SmoothL1Loss()

            min_val_loss = float('inf')
            max_spear_cor = float('-inf')
            no_improve_epochs = 0

            for epoch_idx in range(epochs):
                print("=" * 25)
                print(f"Score Index: {score_index}")
                print(f"noise_std: {noise_std}")
                print("epoch ", epoch_idx)
                avg_train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, sp_weight, score_index)
                print(f"train_loss: {avg_train_loss:.4f}")

                # validation
                val_loss, spear = validation(val_dataloader, model, criterion, score_index)
                scheduler.step(val_loss)
                print("val_loss: ", val_loss, " | spear corr: ", spear)
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    os.makedirs("./fs800_result/auto", exist_ok=True)
                    torch.save(model.state_dict(), f"./fs800_result/auto/checkpoint_pe_score-index-{score_index}_noise-{noise_std}.pth")
                if spear > max_spear_cor:
                    max_spear_cor = spear
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                print("no_improve_epochs: " + str(no_improve_epochs))
                if no_improve_epochs >= no_improvements_tolerance:
                    print(f"Early stopping: no improvement in {no_improvements_tolerance} epochs.")
                    break
                print("min validation loss: ", min_val_loss, " | max spear corr: ", max_spear_cor)
                print("checkpoint_pe")
                print(optimizer.param_groups[0]['lr'])
            print("=" * 25)
            print(f"Report: Score index: {score_index}, Noise std: {noise_std}, min validation loss: {min_val_loss}, max spear corr: {max_spear_cor}")
            result.append({
                "score_index": score_index,
                "noise_std": noise_std,
                "min_val_loss": min_val_loss,
                "max_spear_cor": max_spear_cor,
            })
    print("=" * 25)
    print("Done! Result: ")
    print(result)