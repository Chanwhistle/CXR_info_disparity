import json

with open("/ssd1/chanhwi/long-clinical-doc/dataset/dev_summarization/full-dev-indent-images.json", "r") as f:
    data = json.load(f) 


for sample in data['data']:
    if sample['id'] == "10003502-DS-9":
        print(sample)
breakpoint() 