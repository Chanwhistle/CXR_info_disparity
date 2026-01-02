from datetime import datetime
import pandas as pd
import json


# Load note2hash mapping
with open("/ssd1/chanhwi/long-clinical-doc/dataset/metadata.json", 'r', encoding='utf-8') as f_mapper:
    mapper = json.load(f_mapper)
note2hash = {note_info['note_id']: hash_key for hash_key, note_info in mapper.items()}

# Load image path parser
with open("../dataset/train_summarization/full-train-indent-images.json", 'r', encoding='utf-8') as f_image:
    img_path_parser = json.load(f_image)

hash2meta = {}
for metadata in img_path_parser['data']:
    note_id = metadata['id']
    hash_key = note2hash.get(note_id, None)
    
    # Parse in_time and out_time
    try:
        in_time = pd.to_datetime(metadata['debug_features']['INTIME'])
        out_time = pd.to_datetime(metadata['debug_features']['OUTTIME'])
        assert in_time < out_time, f"INTIME {in_time} must be earlier than OUTTIME {out_time}"
    except Exception as e:
        print(f"Error parsing in_time or out_time for note_id {note_id}: {e}")
        continue

    metadata_filtered = []

    # Process images if present
    for img in metadata.get('images', []):
        # Parse image study date and time
        study_date = datetime.strptime(str(img['StudyDate']), "%Y%m%d")
        study_time = datetime.strptime(f"{int(img['StudyTime']):06}", "%H%M%S").time()
        final_datetime = pd.Timestamp(datetime.combine(study_date, study_time))

        # Filter images within in_time and out_time
        if in_time <= final_datetime <= out_time:
            metadata_filtered.append((
                final_datetime,
                img['path'],
                img['PerformedProcedureStepDescription'],
                img['ViewPosition'],
                img['PatientOrientationCodeSequence_CodeMeaning']
            ))

    # Remove duplicates and sort data_paths by datetime
    metadata_filtered = sorted(set(metadata_filtered), key=lambda x: x[0])

    # Populate hash2meta dictionary
    if hash_key:
        hash2meta[hash_key] = {
            "note_id": note_id,
            "metadata_filtered": metadata_filtered,
        }