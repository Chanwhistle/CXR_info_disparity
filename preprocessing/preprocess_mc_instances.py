"""
MultiChannel 모델용 HDF5 데이터셋을 생성하는 전처리 스크립트.

각 환자의 입원 기간 내 모든 CXR 이미지를 정렬·변환하여 HDF5 파일에 저장합니다.
인스턴스당 이미지 개수가 다를 수 있으므로 variable-length 구조로 저장합니다.

Usage:
    python preprocess_mc_instances.py <input.json> <mimic-cxr-jpg-dir> <output.hdf5>

    <input.json>         : add_images_to_json.py 출력 JSON (이미지 경로 포함)
    <mimic-cxr-jpg-dir>  : MIMIC-CXR-JPG 루트 디렉토리 (512 리사이즈 이미지 포함)
    <output.hdf5>        : 저장할 HDF5 파일 경로

Note: 먼저 resize_jpgs.py로 이미지를 리사이즈해야 합니다.
"""

import sys
import torch
import json
from os.path import join, exists
import logging

from tqdm import tqdm
import imageio
import h5py
import torchvision.transforms as tfms
from PIL import Image

# 이미지 전처리 상수 (train_mortality_xray.py와 동일하게 맞춤)
MIN_RES = 512
MEAN = 0.0
STDEV = 0.229

cxr_train_transforms = tfms.Compose([
    tfms.ToPILImage(),
    tfms.RandomAffine((-5, 5), translate=None, scale=None, shear=(0.9, 1.1)),
    tfms.RandomResizedCrop((MIN_RES, MIN_RES), scale=(0.5, 0.75), ratio=(0.95, 1.05), interpolation=Image.Resampling.LANCZOS),
    tfms.ToTensor(),
    tfms.Normalize((MEAN,), (STDEV,))
])

cxr_infer_transforms = tfms.Compose([
    tfms.ToPILImage(),
    tfms.Resize((MIN_RES, MIN_RES), interpolation=Image.Resampling.LANCZOS),
    tfms.CenterCrop(MIN_RES),
    tfms.ToTensor(),
    tfms.Normalize((MEAN,), (STDEV,))
])


def main(args):
    if len(args) < 2:
        sys.stderr.write("Required argument(s): <json file path> <mimic cxr jpg path> <output filename>\n")
        sys.exit(-1)

    input_json, mimic_path, output_fn = args

    # 파일명에 'train' 포함 여부로 augmentation 결정
    if 'train' in input_json:
        print("Using training transforms since the string 'train' was found in the input file")
        transform = cxr_train_transforms
    else:
        print("Using inference transforms since the string 'train' was not found in the input filename.")
        transform = cxr_infer_transforms

    with open(input_json, 'rt') as fp:
        data_json = json.load(fp)

    num_insts = 0
    for inst in data_json['data']:
        if 'images' in inst.keys() and len(inst['images']) > 0:
            num_insts += 1

    data_paths = []
    labels = {}
    notes = {}
    print("Processing data paths")
    for inst in tqdm(data_json['data']):
        if 'images' not in inst or len(inst['images']) == 0:
            continue

        adm_dt = inst["debug_features"]["ADMITTIME"].split(" ")[0].replace("-","")
        inst_paths = []
        sorted_images = sorted(inst['images'], key=lambda x: x['StudyDate'])
        for image in sorted_images:
            if str(image["StudyDate"]) >= adm_dt:
                img_path = join(mimic_path, 'files', image['path'][:-4] + '_%d_resized.jpg' % (MIN_RES))
                inst_paths.append(img_path)

        # 이미지가 1~7장인 인스턴스만 포함 (과도하게 많은 경우 제외)
        if len(inst_paths) > 0 and len(inst_paths) < 8:
            data_paths.append((inst['id'], inst_paths, inst["debug_features"]["HADM_ID"]))
            labels[inst['id']] = inst['out_hospital_mortality_30']
            notes[inst['id']] = inst['text']

    print("Out of %d instances with images, %d have images from the same encounter" % (num_insts, len(data_paths)))

    print("Compiling multi-channel images for each instance")
    with h5py.File(output_fn, "w") as of:
        of['/len'] = len(data_paths)
        for inst_ind, inst in enumerate(tqdm(data_paths)):
            inst_id, inst_image_paths, inst_hadm = inst
            inst_images = [imageio.imread(img_path, mode='F') for img_path in inst_image_paths]
            inst_images = [transform(image) for image in inst_images]

            of['/%d/data' % inst_ind] = torch.stack(inst_images)
            of['/%d/label' % inst_ind] = labels[inst_id]
            of['/%d/id' % inst_ind] = inst_id
            of['/%d/hadm' % inst_ind] = str(inst_hadm)
            of['/%d/text' % inst_ind] = notes[inst_id]


if __name__ == '__main__':
    main(sys.argv[1:])
