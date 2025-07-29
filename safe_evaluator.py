# safe_evaluator.py

from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import inference_on_dataset, SemSegEvaluator
import logging

def validate_metadata(dataset_name: str):
    metadata = MetadataCatalog.get(dataset_name)
    missing_fields = []

    # Commonly required metadata fields
    required_fields = ["stuff_classes"]

    for field in required_fields:
        if not hasattr(metadata, field):
            missing_fields.append(field)

    return missing_fields

def patch_metadata(dataset_name: str):
    metadata = MetadataCatalog.get(dataset_name)
    patched = False

    if not hasattr(metadata, "stuff_classes") and hasattr(metadata, "thing_classes"):
        metadata.stuff_classes = metadata.thing_classes
        print(f"[safe_evaluator] Patched 'stuff_classes' using 'thing_classes' for dataset '{dataset_name}'")
        patched = True

    return patched

def safe_test(cfg, model, dataset_name: str):
    print(f"[safe_evaluator] Validating metadata for '{dataset_name}'...")

    missing = validate_metadata(dataset_name)
    if missing:
        print(f"[safe_evaluator] Missing metadata fields: {missing}")
        patched = patch_metadata(dataset_name)

        if not patched:
            print(f"[safe_evaluator] Could not patch metadata. Skipping evaluation.")
            return None

    print(f"[safe_evaluator] Running evaluation...")
    evaluator = SemSegEvaluator(dataset_name, distributed=False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, dataset_name)
    return inference_on_dataset(model, val_loader, evaluator)
