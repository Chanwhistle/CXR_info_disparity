"""
단일 CXR 이미지 분류를 위한 Conv-Pool-FC 기반 베이스라인 모델.

입력: 512x512 그레이스케일 이미지 텐서
출력: 2-class 로짓 (0=생존, 1=사망)

Source: Modified from LCD Benchmark (https://github.com/Machine-Learning-for-Medical-Language/long-clinical-doc)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineMortalityPredictor(nn.Module):
    def __init__(self, shape, num_filters=128, kernel_size=5, pool_size=50):
        super(BaselineMortalityPredictor, self).__init__()
        stride = 1
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size)
        self.hout_conv = int(torch.floor(torch.tensor(1 + (shape[0] - (kernel_size-1) - 1) / stride)).item())
        self.wout_conv = int(torch.floor(torch.tensor(1 + (shape[1] - (kernel_size-1) - 1) / stride)).item())

        self.pool = nn.MaxPool2d(pool_size)
        pool_stride = pool_size
        self.hout_pool = int(torch.floor(torch.tensor(1 + (self.hout_conv - (pool_size-1) - 1) / pool_stride)).item())
        self.wout_pool = int(torch.floor(torch.tensor(1 + (self.wout_conv - (pool_size-1) - 1) / pool_stride)).item())

        self.fc1 = nn.Linear(num_filters * self.hout_pool * self.wout_pool, 2)
        print("Model initialized with hidden layer containing %d nodes" % (self.fc1.in_features))

    def forward(self, matrix):
        unpooled = F.relu(self.conv1(matrix))
        pooled = self.pool(unpooled)
        x = pooled.view(matrix.shape[0], -1)
        out = self.fc1(x)
        return out
