import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import sys
import os
from yacs.config import CfgNode as CN
import torchvision.transforms as transforms
import json
from collections import Counter
import numpy as np

# This package internal functions should be used here
from softlabel_od.adapters.dlcv.config import get_cfg_defaults, get_cfg_from_file
from softlabel_od.adapters.dlcv.dataset import *
from softlabel_od.adapters.dlcv.utils import *
from softlabel_od.adapters.dlcv.training import train_one_epoch
from softlabel_od.adapters.dlcv.training_1d import train_and_evaluate_model_1d
from softlabel_od.adapters.dlcv.training_2d import train_and_evaluate_model_2d
from softlabel_od.adapters.dlcv.model import get_model
from softlabel_od.adapters.dlcv.augmentations import ClassConditionalAugmentation, MinimalAugmentation

def sum(a, b):
    x=a+b
    return x

def main(cfg):
    # Determine which soft label file to use based on GMM type
    if cfg.DATA.GMM_TYPE == 'HARD_LABEL':
        dataset_type = 'hard'
        label_file = cfg.DATA.HARD_LABEL_FILE
        gmm_type = None  # Not used for hard labels
    else:
        dataset_type = 'soft'
        gmm_type = cfg.DATA.GMM_TYPE
        # Choose soft label file
        if cfg.DATA.GMM_TYPE == '1D':
            label_file = cfg.DATA.SOFT_LABEL_FILE_1D
        elif cfg.DATA.GMM_TYPE == '2D':
            label_file = cfg.DATA.SOFT_LABEL_FILE_2D
        else:
            raise ValueError(f"Unsupported GMM type: {cfg.DATA.GMM_TYPE}")

    if not os.path.exists(cfg.DATA.ROOT):
        print(f"Dataset root path does not exist: {cfg.DATA.ROOT}")
    else:
        print(f"Dataset root path exists: {cfg.DATA.ROOT}")
        
    if not os.path.exists(label_file):
        print(f"Label file does not exist: {label_file}")
    else:
        print(f"Label file exists: {label_file}")

    print(f"Using configuration file: {getattr(cfg, 'CONFIG_FILE_PATH', 'N/A')}")
    print(f"Using {cfg.DATA.GMM_TYPE} GMM soft labels")
    print("Configuration for this run:")
    print(cfg.dump())

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.MISC.NO_CUDA else "cpu")
    print(f'Device: {device}')

    # Define transformations for training and testing
    rare_class_indices = cfg.AUGMENTATION.RARE_CLASS_INDICES

    train_transform = ClassConditionalAugmentation(
    rare_class_indices=rare_class_indices,
    rare_augment_prob=cfg.AUGMENTATION.RARE_CLASS_AUGMENT_PROB,      
    common_augment_prob=cfg.AUGMENTATION.COMMON_CLASS_AUGMENT_PROB,     
    rotation_degrees=cfg.AUGMENTATION.ROTATION_DEGREES,
    brightness_range=cfg.AUGMENTATION.BRIGHTNESS_RANGE,  
    contrast_range=cfg.AUGMENTATION.CONTRAST_RANGE,    
    noise_std=cfg.AUGMENTATION.NOISE_STD                
)

    test_transform = MinimalAugmentation()

    # Load datasets with proper parameters
    print("Initializing training dataset...")
    train_dataset = Vin_Dr_CXR_SoftLabel_Dataset(
        root=cfg.DATA.ROOT, 
        label_file=label_file,
        transforms=train_transform, 
        train=True,
        test_split_ratio=cfg.DATA.TEST_SPLIT_RATIO,
        gmm_type=gmm_type,
        dataset_type=dataset_type
    )
    
    print("Initializing validation dataset...")
    test_dataset = Vin_Dr_CXR_SoftLabel_Dataset(
        root=cfg.DATA.ROOT, 
        label_file=label_file,
        transforms=test_transform, 
        train=False,
        test_split_ratio=cfg.DATA.TEST_SPLIT_RATIO,
        gmm_type=gmm_type,
        dataset_type=dataset_type
    )

    
    # Initialize training and test loader
    if cfg.DATA.BALANCED_SAMPLER:
        print("Using balanced sampling for training...")
        sampler = create_balanced_sampler(
            train_dataset, 
            rfs_threshold=cfg.DATA.OVERSAMPLE_FACTOR,
            # presence_prob_thresh=cfg.DATA.PRESENCE_PROB_THRESH,
            ignore_nofinding=cfg.DATA.NOFINDING_WEIGHT_GIVEN
        )
        train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.TRAIN.BATCH_SIZE, 
        sampler=sampler, 
        collate_fn=collate_fn, 
        num_workers=4,
        pin_memory=True,              
        persistent_workers=True,      
        prefetch_factor=2,           
        )
    else:
        print("Using standard sampling...")
        
        # --- Manual filtering for standard sampler ---
        if cfg.DATA.IGNORE_NOFINDING:
            print("Filtering No-Finding images (Class 0)...")
            valid_indices = []
            
            # Iterate through dataset to find valid images
            for idx in range(len(train_dataset)):
                img_id = train_dataset.image_ids[idx]
                labels = train_dataset.image_data[img_id]
                dominant_class = train_dataset.get_dominant_class(labels)
                
                # Keep if it is NOT class 0 (assuming 0 is No Finding)
                if dominant_class != 0:
                    valid_indices.append(idx)
            
            print(f"Kept {len(valid_indices)}/{len(train_dataset)} images.")
            
            # Use SubsetRandomSampler to restrict training to these indices
            sampler = torch.utils.data.SubsetRandomSampler(valid_indices)
            shuffle = False # SubsetRandomSampler already shuffles
        else:
            # Standard random sampling of everything
            sampler = None
            shuffle = True

        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.TRAIN.BATCH_SIZE, 
            shuffle=shuffle, # Set by logic above
            sampler=sampler, # Set by logic above
            collate_fn=collate_fn, 
            num_workers=4,
            pin_memory=True,              
            persistent_workers=True,      
            prefetch_factor=2,           
        )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.TRAIN.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=4
    )

    # Handle overfit mode
    overfit_batches = getattr(cfg.TRAIN, 'OVERFIT_BATCHES', 0)
    if overfit_batches > 0:
        from softlabel_od.adapters.dlcv.utils import OverfitDataLoader
        print(f"\n{'='*60}")
        print(f"⚠️  OVERFIT MODE ENABLED - {overfit_batches} batches")
        print(f"{'='*60}")
        
        train_loader = OverfitDataLoader(
            train_loader, 
            num_batches=overfit_batches,
            move_to_device=device
        )
        
        # Use SAME batches for validation (true overfit test)
        test_loader = train_loader
        
        print(f"📊 Train and Val use SAME {len(train_loader)} cached batches")
        print("\n🔍 OVERFIT DATASET INSPECTION:")
        # Iterate through the cached batches in the loader
        for batch_idx, (images, targets) in enumerate(train_loader):
            print(f"  Batch {batch_idx+1}:")
            for i, t in enumerate(targets):
                img_id = t.get('image_id', 'Unknown')
                # Handle tensor image_id
                if isinstance(img_id, torch.Tensor):
                    img_id = img_id.item()
                
                print(f"    Img {i+1} (ID: {img_id}):")
                
                # Boxes
                if len(t['boxes']) > 0:
                    for box, label in zip(t['boxes'], t['labels']):
                        # Check if we have means (soft label) or just boxes (hard label)
                        if 'gt_means' in t:
                            print(f"      - Box: {box.tolist()} | Label: {label.item()}")
                        else:
                            print(f"      - Box: {box.tolist()} | Label: {label.item()}")
                else:
                    print("      - No Boxes (No Finding)")
                    
                # Class Probs (if available)
                if 'gt_class_probs' in t and len(t['gt_class_probs']) > 0:
                    probs = t['gt_class_probs'][0]
                    # Find dominant class
                    dom_idx = torch.argmax(probs).item()
                    dom_prob = probs[dom_idx].item()
                    print(f"      - Dominant Class: {dom_idx} ({dom_prob:.2f})")
        print("="*60 + "\n")

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(test_dataset)}")

    # Initialize model
    if cfg.DATA.GMM_TYPE == '1D':
        model_type = '1d'
        from softlabel_od.adapters.dlcv.training_1d import train_and_evaluate_model_1d as train_fn
    elif cfg.DATA.GMM_TYPE == '2D':
        model_type = '2d'
        from softlabel_od.adapters.dlcv.training_2d import train_and_evaluate_model_2d as train_fn
    elif cfg.DATA.GMM_TYPE == 'HARD_LABEL':
        model_type = 'hard'
        from softlabel_od.adapters.dlcv.training_hard_label import train_and_evaluate_model_hard as train_fn
    else:
        raise ValueError(f"Unsupported GMM type: {cfg.DATA.GMM_TYPE}")
    
    # Get model using factory
    model = get_model(
        num_classes=cfg.MODEL.NUM_CLASSES,
        model_type=model_type,
        backbone_name=cfg.MODEL.BACKBONE,
        pretrained=True,
        probabilistic=cfg.MODEL.PROBABILISTIC
    )
    model.to(device)

     # Load pretrained weights if specified
    if cfg.MISC.PRETRAINED_WEIGHTS:
        model = load_pretrained_weights(model, cfg.MISC.PRETRAINED_WEIGHTS, device)

    # Compute class weights for the ROI head
    class_weights = train_dataset.compute_roi_head_class_weights_per_cluster(
        num_classes=cfg.MODEL.NUM_CLASSES,
        presence_prob_thresh=0.0,
        normalize=False,
        verbose=True,
    )
    model.roi_heads.register_buffer("class_weights", class_weights.to(device))

   

    # Freeze layers if set as argument
    if cfg.MISC.FROZEN_LAYERS:
        freeze_layers(model, cfg.MISC.FROZEN_LAYERS)
        print(f"Frozen layers: {cfg.MISC.FROZEN_LAYERS}")

    # Define the optimizer
    bbox_params = []
    class_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Skip frozen layers
        if 'box_predictor.bbox_pred' in name:
            bbox_params.append(param)
        elif 'box_predictor.cls_score' in name:
            class_params.append(param)
        else:
            other_params.append(param)

    # params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD([
        {'params': bbox_params, 'lr': cfg.TRAIN.BASE_LR_BBOX},  # Bbox
        {'params': class_params, 'lr': cfg.TRAIN.BASE_LR_CLASS}, # Class
        {'params': other_params, 'lr': cfg.TRAIN.BASE_LR},      # Rest of the model
    ], momentum=0.9, weight_decay=0.001)

    # Main learning rate scheduler
    main_scheduler = MultiStepLR(
        optimizer, milestones=cfg.TRAIN.MILESTONES, gamma=cfg.TRAIN.GAMMA
    )
    
    secondary_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.TRAIN.NUM_EPOCHS,      # full cycle over training
        eta_min=1e-5           # don't go below this
    )
    if cfg.TRAIN.SCHEDULER == "main_scheduler":
        scheduler = main_scheduler
    elif cfg.TRAIN.SCHEDULER == "secondary_scheduler":
        scheduler = secondary_scheduler
    else:
        raise ValueError(f"Unsupported scheduler: {cfg.TRAIN.SCHEDULER}")

    # Create output directories
    os.makedirs(cfg.MISC.SAVE_MODEL_PATH, exist_ok=True)
    os.makedirs(cfg.MISC.RESULTS_CSV, exist_ok=True)

    # Hand everything to the train and evaluate model function
    train_fn(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        num_epochs=cfg.TRAIN.NUM_EPOCHS,
        device=device,
        early_stopping=cfg.TRAIN.EARLY_STOPPING,
        scheduler=scheduler,
        config_name=cfg.MISC.RUN_NAME
    )
    # Save the model using the default folder
    if cfg.MISC.SAVE_MODEL_PATH:
        save_model(model, cfg.MISC.SAVE_MODEL_PATH + "/" + cfg.MISC.RUN_NAME)
        print(f"Model saved to: {cfg.MISC.SAVE_MODEL_PATH}/{cfg.MISC.RUN_NAME}")
    
    # Save the configuration used for this run
    config_save_path = os.path.join(cfg.MISC.SAVE_MODEL_PATH, cfg.MISC.RUN_NAME + '_runConfig.yaml')
    with open(config_save_path, 'w') as f:
        f.write(cfg.dump())
    print(f"Configuration saved to: {config_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a VinDr-CXR soft label model")
    parser.add_argument('--config', type=str, default='src/dlcv/configs/vindr_cxr_hard_label_linux.yaml', help="Path to the config file")
    args = parser.parse_args()
    
    cfg = get_cfg_defaults()

    # Check if the script is run with the command-line argument or from the notebook
    if args.config:
        config_file_path = args.config
    else:
        # Fallback to an environment variable or a default config path
        config_file_path = os.getenv('CONFIG_FILE', 'configs/config_first.yaml')

    cfg.CONFIG_FILE_PATH = config_file_path
    # cfg.merge_from_file(config_file_path)
    cfg = get_cfg_from_file(config_file_path)
    main(cfg)
