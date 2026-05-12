"""
Facebook DINOv2-small 기반 CXR 분류 모델 (단일 이미지).

Source: Modified from LCD Benchmark (https://github.com/Machine-Learning-for-Medical-Language/long-clinical-doc)
"""

import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor


class VitMortalityPredictor(nn.Module):
    def __init__(self, shape, num_filters=64, kernel_size=5, pool_size=50):
        super(VitMortalityPredictor, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        self.encoder = AutoModel.from_pretrained('facebook/dinov2-small')
        # 분류 헤드는 별도로 정의 필요 (encoder 출력 dim 확인 후 추가)
        # self.fc = nn.Linear(encoder_output_dim, 2)

    def forward(self, matrix):
        output = self.encoder(matrix)
        logits = self.fc(output['pooler_output'])
        return logits
