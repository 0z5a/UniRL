import os

import numpy as np
import torch


def validate_video_csv_files(args, logger):
    """
    Validate that all video_csv files exist before starting training.
    
    Args:
        args: Training arguments object containing video_csv attribute
        logger: Logger instance for outputting validation results
    
    Raises:
        FileNotFoundError: If any video_csv files are missing
    """
    if not hasattr(args, 'video_csv') or args.video_csv is None:
        logger.info("No video_csv files specified, skipping validation.")
        return
    
    logger.info("=" * 80)
    logger.info("VALIDATING VIDEO_CSV FILES")
    logger.info("=" * 80)
    
    missing_files = []
    valid_files = []
    
    for i, csv_path in enumerate(args.video_csv):
        if os.path.exists(csv_path):
            valid_files.append(csv_path)
            logger.info(f"✓ [{i+1:2d}/{len(args.video_csv)}] Found: {csv_path}")
        else:
            missing_files.append(csv_path)
            logger.error(f"✗ [{i+1:2d}/{len(args.video_csv)}] Missing: {csv_path}")
    
    logger.info("-" * 80)
    logger.info(f"Total video_csv files: {len(args.video_csv)}")
    logger.info(f"Valid files: {len(valid_files)}")
    logger.info(f"Missing files: {len(missing_files)}")
    
    if missing_files:
        logger.error("=" * 80)
        logger.error("VALIDATION FAILED - MISSING FILES:")
        for missing_file in missing_files:
            logger.error(f"  - {missing_file}")
        logger.error("=" * 80)
        logger.error("Please check the file paths in your configuration and ensure all files exist.")
        logger.error("Training cannot continue with missing video_csv files.")
        raise FileNotFoundError(f"Missing {len(missing_files)} video_csv files. See log for details.")
    else:
        logger.info("✓ All video_csv files validated successfully!")
        logger.info("=" * 80)

def convert_to_json_serializable(obj):
    """Convert NumPy and other non-JSON-serializable types to native Python types."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    else:
        return obj
