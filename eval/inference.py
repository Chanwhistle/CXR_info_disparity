"""
Run mortality inference from a fine-tuned checkpoint.

Example:
    python -m eval.inference \
        --use_discharge_note \
        --checkpoint_dir ../trained_models/dn_only \
        --base_img_dir ../saved_images_560 \
        --base_rr_dir ../physionet.org/files/mimic-cxr/2.1.0/files \
        --output_path ../results/dn_only
"""

import json
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from core.models import load_model
from core.utils import get_args, log_result, prepare_loader, set_seed


def inference(model, device, data_loader, args, set_type):
    """Run inference and save predictions."""
    output_dict = {"data": []}
    predictions_file = os.path.join(
        args.output_path,
        f'{args.summary_type}{"_add_pi" if args.use_pi else ""}_{set_type}_predictions.txt'
    )
    os.makedirs(os.path.dirname(predictions_file), exist_ok=True)

    with open(predictions_file, 'w') as f:
        for batch in tqdm(data_loader, total=len(data_loader)):
            batch_device = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                outputs = model(**batch_device)

            logits = outputs["logits"]
            probabilities = F.softmax(logits.float(), dim=-1)

            for i in range(len(probabilities)):
                output = {
                    "id": batch["ids"][i],
                    "label": int(batch["labels"][i].item()) if "labels" in batch else None,
                    "logits": logits[i].cpu().tolist(),
                    "probabilities": probabilities[i].cpu().tolist()
                }
                f.write(json.dumps(output) + "\n")
                output_dict["data"].append(output)

    labels, probs = zip(*[(ele['label'], ele['probabilities']) for ele in output_dict['data']])
    log_result(args, labels, probs, args.output_path, set_type)


def test(args):
    model, processor = load_model(args, model_id=args.model_name_or_path, inference=True)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.config.use_cache = True

    test_loader = prepare_loader(args.summary_type, args, "test", processor)
    inference(model, device, test_loader, args, set_type="test")


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    test(args)
