"""
복수 CXR 이미지를 ViT-L/16 + Transformer Encoder로 집계하는 MultiChannel 모델.

입원 기간 내 모든 CXR을 순서대로 인코딩하고, learnable CLS 토큰으로 최종 예측을 생성합니다.
HDF5 포맷의 variable-length 이미지 시퀀스 데이터셋 클래스도 포함합니다.

Source: Modified from LCD Benchmark (https://github.com/Machine-Learning-for-Medical-Language/long-clinical-doc)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import imageio
from tqdm import tqdm
import h5py
import torchvision.models as models


class MultiChannelMortalityPredictor(nn.Module):
    def __init__(self, shape, embed_dim=1024, num_heads=16):
        super(MultiChannelMortalityPredictor, self).__init__()
        # ViT-L/16으로 각 이미지 인코딩 (출력 dim=1024)
        self.encoder = models.vision_transformer.vit_l_16(models.vision_transformer.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        self.encoder.heads.head = nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # 시퀀스 앞에 붙이는 learnable CLS 토큰
        self.cls = torch.rand(embed_dim)
        self.fc1 = nn.Linear(embed_dim, 2)

    def forward(self, batch_input, output_hidden_states=False):
        if not self.cls.device == batch_input.device:
            self.cls = self.cls.to(batch_input.device)

        batch_size = batch_input.shape[0]
        # 그레이스케일 → RGB (3채널 복제)
        rgb_matrix = torch.repeat_interleave(batch_input, 3, dim=1)

        channels = rgb_matrix.shape[2]
        input_reps = torch.stack([self.encoder(rgb_matrix[:, :, i, :, :]) for i in range(channels)])
        input_reps = torch.cat([
            torch.repeat_interleave(self.cls.unsqueeze(0).unsqueeze(0), batch_size, dim=1),
            input_reps
        ], dim=0)

        output_reps = self.transformer_encoder(input_reps)

        # CLS 토큰 위치(index 0)로 분류
        out = self.fc1(output_reps[0])
        if output_hidden_states:
            return out, output_reps[0]
        else:
            return out


class VariableLengthImageDataset(Dataset):
    """HDF5 파일에서 variable-length 이미지 시퀀스를 로드하는 데이터셋."""

    def __init__(self, hdf5_fn):
        self.fp = h5py.File(hdf5_fn, "r")
        self.labels = []
        for idx in range(len(self)):
            self.labels.append(self.fp['%d/label' % (idx)][()])

    def __len__(self):
        return self.fp['/len/'][()]

    def __getitem__(self, idx):
        return torch.tensor(self.fp['%d/data' % (idx)]), self.get_text(idx), self.labels[idx]

    def get_metadata(self, idx):
        return self.fp['%d/hadm' % (idx)][()].decode()

    def get_id(self, idx):
        return self.fp['%d/id' % (idx)][()].decode()

    def get_text(self, idx):
        return self.fp['/%d/text' % (idx)][()].decode()


def collate_fn(batch):
    """Variable-length 이미지 배치를 zero-padding으로 맞춤."""
    images_batch, text_batch, labels_batch = zip(*batch)
    max_length = max(len(image_list) for image_list in images_batch)
    padded_batch = []

    for image_list in images_batch:
        padding = torch.zeros((max_length - image_list.shape[0], 1, image_list.shape[2], image_list.shape[3]))
        padded_batch.append(torch.cat([image_list, padding]))

    # batch_size=1일 때 squeeze가 배치 차원을 없애지 않도록 dim=2 명시
    return torch.squeeze(torch.stack(padded_batch), dim=2), text_batch, torch.tensor(labels_batch)
