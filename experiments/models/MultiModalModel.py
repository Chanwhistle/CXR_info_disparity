"""
임상 텍스트(CNN)와 CXR 이미지(ViT/ResNet)를 융합하는 멀티모달 분류 모델.

텍스트: cnlpt 라이브러리의 CnnSentenceClassifier (출력 dim=1500)
이미지: 외부에서 학습된 ViT/ResNet 모델 (출력 dim=1024)
두 표현을 concatenate 후 FC 레이어로 최종 분류.

Source: Modified from LCD Benchmark (https://github.com/Machine-Learning-for-Medical-Language/long-clinical-doc)
"""

import torch.nn as nn
import torch
from cnlpt.BaselineModels import CnnSentenceClassifier
from os.path import join
import json
from transformers import AutoTokenizer


class MultiModalMortalityPredictor(nn.Module):
    def __init__(self, text_model=None, img_model=None):
        super(MultiModalMortalityPredictor, self).__init__()

        conf_file = join(text_model, "config.json")
        with open(conf_file, "rt") as fp:
            conf_dict = json.load(fp)

        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        num_labels_dict = {'mortality': 2}
        self.text_model = CnnSentenceClassifier(
            len(self.tokenizer),
            task_names=['mortality'],
            num_labels_dict=num_labels_dict,
            embed_dims=conf_dict["cnn_embed_dim"],
            num_filters=conf_dict["cnn_num_filters"],
            filters=conf_dict["cnn_filter_sizes"],
        )
        self.text_model.load_state_dict(torch.load(join(text_model, "pytorch_model.bin")))

        self.img_model = torch.load(img_model)
        # 텍스트 CNN 출력 1500 + ViT 이미지 출력 1024
        self.fc = nn.Linear(1500 + 1024, 2)

    def forward(self, img_matrix, text):
        max_len = max([len(x) for x in text])
        text_encoding = self.tokenizer(text, max_length=max_len, truncation=True, padding='max_length')
        input_ids = torch.LongTensor(text_encoding['input_ids'])
        _, _, text_rep = self.text_model(input_ids.to(img_matrix.device), output_hidden_states=True)
        _, img_rep = self.img_model(img_matrix, output_hidden_states=True)

        mm_rep = torch.cat([text_rep, img_rep], dim=1)
        logits = self.fc(mm_rep)
        return logits
