"""
MIMIC-CXR-JPG 디렉토리 내 이미지 해상도 분포를 출력하는 유틸리티 스크립트.

Usage:
    python extract_image_sizes.py <input-directory>

    <input-directory> : JPG 이미지가 있는 디렉토리 (재귀 탐색)

Source: Modified from LCD Benchmark (https://github.com/Machine-Learning-for-Medical-Language/long-clinical-doc)
"""

import os, sys
from PIL import Image
from os.path import join


def main(args):
    if len(args) < 1:
        sys.stderr.write("Required argument(s): <input directory>\n")
        sys.exit(-1)

    sizes = dict()
    for root, dirs, files in os.walk(args[0]):
        for file in files:
            if file.endswith(".jpg"):
                img = Image.open(join(root, file))
                size_str = ",".join([str(x) for x in img.size])
                if size_str not in sizes:
                    sizes[size_str] = 0
                    print("New size string found: %s" % (size_str))
                sizes[size_str] += 1

    for size, count in sizes.items():
        print("%s => %d" % (size, count))


if __name__ == "__main__":
    main(sys.argv[1:])
