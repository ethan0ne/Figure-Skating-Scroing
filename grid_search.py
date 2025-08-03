from typing_extensions import final
import torch
import torch.utils.data as data
import os
import numpy as np
from model import scoring_head
from dataset.dataset_fs1000 import FeatureDataset, av_collate_fn
from scipy.stats import spearmanr
import math
import random
import torch.nn.functional as F
import itertools
import logging

logging.basicConfig(
    filename='grid_search.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s %(message)s'
)

def pearson_loss(pred, target):
    vx = pred - pred.mean()
    vy = target - target.mean()
    corr = (vx * vy).sum() / (torch.sqrt((vx**2).sum()) * torch.sqrt((vy**2).sum()) + 1e-8)
    return 1 - corr

# Spearman loss using approximate differentiable ranks
def spearman_loss(pred, target, scale=1.0):
    pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
    pred_ranks = torch.sigmoid(-scale * pred_diff).sum(dim=1)
    targ_diff = target.unsqueeze(1) - target.unsqueeze(0)
    targ_ranks = torch.sigmoid(-scale * targ_diff).sum(dim=1)
    vx = pred_ranks - pred_ranks.mean()
    vy = targ_ranks - targ_ranks.mean()
    corr = (vx * vy).sum() / (torch.sqrt((vx**2).sum()) * torch.sqrt((vy**2).sum()) + 1e-8)
    return 1 - corr

# Set fixed seed for reproducibility
seed = 2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# detect MPS (Apple Silicon) or fallback to CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def validation(dataloader, model, criterion, score_index):
    model.eval()
    val_loss = 0
    val_truth = []
    val_pred = []

    for audio_feature, video_feature, inv_audio_feature, inv_video_feature, audio_len, video_len, score, data_index in dataloader:
        batch_size = audio_feature.size(0)
        audio_feature = audio_feature.to(device)
        video_feature = video_feature.to(device)
        inv_audio_feature = inv_audio_feature.to(device)
        inv_video_feature = inv_video_feature.to(device)
        target = score[score_index].to(device)

        with torch.no_grad():
            output = model(audio_feature, video_feature, inv_audio_feature, inv_video_feature, audio_len, video_len)
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
    # parameter grid for grid search
    param_grid = {
        'lr': [1e-2, 1e-3, 5e-4, 1e-4],
        'weight_decay': [1e-4, 5e-5, 1e-5, 1e-6],
        'batch_size': [8, 16, 32]
    }
    best_params = None
    best_spear = float('-inf')
    epochs = 500
    score_index = 4
    patience = 30

    FS1000_dataset_dir = '/Volumes/Data (Samsung PSSD T7 - 2 TB of Ethan)/Study/KCL/MSc Individual Project/References/FS1000 Dataset/'

    # prepare datasets once
    train_dataset = FeatureDataset(root_path=FS1000_dataset_dir, is_train=True)
    val_dataset = FeatureDataset(root_path=FS1000_dataset_dir, is_train=False)

    # grid search
    for lr, wd, bs in itertools.product(
        param_grid['lr'], param_grid['weight_decay'], param_grid['batch_size']
    ):
        logging.info(f'Testing lr={lr}, weight_decay={wd}, batch_size={bs}')
        # 本组超参的早停追踪
        combo_best_spear = float('-inf')
        no_improve_epochs = 0

        # prepare dataloaders
        train_dataloader = data.DataLoader(
            dataset=train_dataset, batch_size=bs, shuffle=True,
            num_workers=0, collate_fn=av_collate_fn
        )
        val_dataloader = data.DataLoader(
            dataset=val_dataset, batch_size=bs,
            num_workers=0, collate_fn=av_collate_fn
        )

        # model & optimizer setup
        model = scoring_head(
            depth=2, input_dim=768, dim=512,
            input_len=16, num_scores=1, bidirection=True
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        criterion = torch.nn.MSELoss()

        # training and validation loop
        for epoch_idx in range(epochs):
            logging.info("=" * 25)
            logging.info(f"lr={lr}, wd={wd}, epoch {epoch_idx}")

            # train
            model.train()
            train_loss_sum = 0.0
            batch_count = 0
            for audio_feat, video_feat, inv_audio_feat, inv_video_feat, audio_len, video_len, score, idx in train_dataloader:
                audio_feat, video_feat = audio_feat.to(device), video_feat.to(device)
                inv_audio_feat, inv_video_feat = inv_audio_feat.to(device), inv_video_feat.to(device)
                target = score[score_index].to(device)

                optimizer.zero_grad()
                output = model(audio_feat, video_feat, inv_audio_feat, inv_video_feat, audio_len, video_len)
                loss_mse = criterion(output, target)
                loss_sp = spearman_loss(output, target)
                loss = loss_mse + 0.1 * loss_sp
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()
                batch_count += 1

            avg_train_loss = train_loss_sum / batch_count
            logging.info(f"train_loss: {avg_train_loss:.4f}")

            # validation
            val_loss, spear = validation(val_dataloader, model, criterion, score_index)
            scheduler.step(spear)
            logging.info(f"val_loss: {val_loss:.4f} | spear corr: {spear:.4f}")

            # 早停检查
            if spear > combo_best_spear:
                combo_best_spear = spear
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= patience:
                logging.info(
                    f"Early stopping for lr={lr}, wd={wd} at epoch {epoch_idx}"
                )
                break

        logging.info(f'Finished lr={lr}, wd={wd} => Best Spearman: {combo_best_spear:.4f}')
        if combo_best_spear > best_spear:
            best_spear = combo_best_spear
            best_params = {'lr': lr, 'weight_decay': wd}

    logging.info(f"Best hyperparameters: {best_params} with Spearman correlation: {best_spear:.4f}")
    print('Grid search complete. See grid_search.log for details.')