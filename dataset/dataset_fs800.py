import torch
import os
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    def __init__(self,
                 root_path,
                 is_train=True,
                 ):

        self.root_path = root_path
        self.feature_dir = os.path.join(root_path, 'new feature')
        list_file = 'train_fs800.txt' if is_train else 'val_fs800.txt'
        file_path = os.path.join(root_path, list_file)
        with open(file_path, 'r') as f:
            self.total_data = [line.strip().split() for line in f]


    def __getitem__(self, index):
        entry = self.total_data[index]
        data_index = entry[0]
        audio_path = os.path.join(self.feature_dir, 'ast_feature_fs1000_new', f'{data_index}.npy')
        video_path = os.path.join(self.feature_dir, 'c3d_feat_fs1000_new', f'{data_index}.npy')
        audio_feature = torch.from_numpy(np.load(audio_path))
        video_feature = torch.from_numpy(np.load(video_path))
        scores = list(map(float, entry[1:8]))  # tes, pcs, ss, trans, perform, composition, interpretation
        factor = float(entry[8])
        scores[1] /= factor  # normalize pcs
        return audio_feature, video_feature, *scores, data_index

    def __len__(self):
        return len(self.total_data)


def av_collate_fn(batch):
    audios = [item[0] for item in batch]
    videos = [item[1] for item in batch]
    inv_audios = [torch.flip(item[0], [0]) for item in batch]
    inv_videos = [torch.flip(item[1], [0]) for item in batch]
    # original scores: item[2] through item[8]
    scores_list = [item[2:9] for item in batch]
    data_index = [item[9] for item in batch]
    audio_len = [a.shape[0] for a in audios]
    video_len = [v.shape[0] for v in videos]

    audios = pad_sequence(audios, batch_first=True).unsqueeze(2)
    videos = pad_sequence(videos, batch_first=True)
    inv_audios = pad_sequence(inv_audios, batch_first=True).unsqueeze(2)
    inv_videos = pad_sequence(inv_videos, batch_first=True)

    # transpose and pad lengths as needed
    scores = [torch.tensor(vals, dtype=torch.float32) for vals in zip(*scores_list)]

    return audios, videos, inv_audios, inv_videos, audio_len, video_len, scores, data_index