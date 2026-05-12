"""
LCD Benchmark JSON에서 CXR 이미지 유형(PerformedProcedureStepDescription) 분포를 출력하는 유틸리티.

add_images_to_json.py 실행 후, 어떤 촬영 유형이 몇 건 존재하는지 확인할 때 사용합니다.

Usage:
    python extract_image_types.py <input.json>

    <input.json> : add_images_to_json.py 출력 JSON (이미지 메타데이터 포함)

Source: Modified from LCD Benchmark (https://github.com/Machine-Learning-for-Medical-Language/long-clinical-doc)
"""

import sys
import json


def main(args):
    if len(args) < 1:
        sys.stderr.write("Required argument(s): <input file (json)>\n")
        sys.exit(-1)

    with open(args[0], 'rt') as fp:
        data = json.load(fp)
        dataset_types = dict()
        inst_counts = 0
        for inst in data['data']:
            if 'images' not in inst or len(inst['images']) == 0:
                continue
            inst_counts += 1
            inst_types = set()
            adm_dt = inst["debug_features"]["ADMITTIME"].split(" ")[0].replace("-","")
            sorted_images = sorted(inst['images'], key=lambda x: x['StudyDate'])
            for image in sorted_images:
                if str(image["StudyDate"]) >= adm_dt:
                    img_type = image["PerformedProcedureStepDescription"]
                    inst_types.add(img_type)

            for img_type in inst_types:
                if img_type not in dataset_types:
                    dataset_types[img_type] = 0
                dataset_types[img_type] += 1

    for key, val in dataset_types.items():
        print("%s => %d" % (key, val))

    print("Total number of instances with images: %d" % (inst_counts))


if __name__ == "__main__":
    main(sys.argv[1:])
