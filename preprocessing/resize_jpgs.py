"""
MIMIC-CXR-JPG 이미지를 512px 기준으로 리사이즈하는 전처리 스크립트.

원본 JPG 이미지를 MIN_RES(기본 512) 기준으로 aspect ratio를 유지하면서 리사이즈하고,
<original_name>_512_resized.jpg 형태로 저장합니다.

Usage:
    python resize_jpgs.py <mimic-cxr-jpg-directory>

    <mimic-cxr-jpg-directory> : MIMIC-CXR-JPG 루트 디렉토리 (재귀적으로 하위 탐색)

Adapted from pytorch-cxr (https://github.com/jinserk/pytorch-cxr)
Source: Modified from LCD Benchmark (https://github.com/Machine-Learning-for-Medical-Language/long-clinical-doc)
"""

import os, sys
from PIL import Image
from os.path import join
from tqdm import tqdm

# 리사이즈 기준 최소 해상도 (짧은 변 기준)
MIN_RES = 512

def main(args):
    if len(args) < 1:
        sys.stderr.write("Required argument(s): <mimic-cxr-jpg directory>\n")
        sys.exit(-1)

    with tqdm(total=371920) as pbar:
        for root, dirs, files in os.walk(args[0]):
            for file in files:
                if file.endswith(".jpg") and not file.endswith("_resized.jpg"):
                    ff = join(root, file)
                    img = Image.open(ff)
                    w, h = img.size
                    # 짧은 변을 MIN_RES에 맞추고 비율 유지
                    rs = (MIN_RES, int(h/w*MIN_RES)) if w < h else (int(w/h*MIN_RES), MIN_RES)
                    resized = img.resize(rs, Image.LANCZOS)
                    out_fn = file[:-4] + "_%d_resized.jpg" % (MIN_RES)
                    out_path = join(root, out_fn)
                    resized.save(out_path, "JPEG", quality=95)
                    pbar.update(1)

if __name__ == "__main__":
    main(sys.argv[1:])
