from yacs.config import CfgNode as CN
import yaml
import os
project_root_ext = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values for the VinDr-CXR project.
    Returns:
        CfgNode: Default configuration object.
    """
    _C = CN()

    # Data settings
    _C.DATA = CN()
    _C.DATA.DATASET = 'VinDr-CXR'
    _C.DATA.ROOT = './data/vinbigdata'
    _C.DATA.SOFT_LABEL_FILE_1D = os.path.join(project_root_ext, 'outputs/vindrcxr/1d_gaussian_softlabels/1DGMM_softLabels_iou0_5_with_priors.json')
    _C.DATA.SOFT_LABEL_FILE_2D = os.path.join(project_root_ext, 'outputs/vindrcxr/2d_gaussian_softlabels/2DGMM_softLabels_iou0_5_with_priors.json')
    _C.DATA.HARD_LABEL_FILE = os.path.join(project_root_ext, 'outputs/vindrcxr/hard_labels/HardLabels_MajorityVote_iou0_5.json')
    _C.DATA.GMM_TYPE = '1D'  # Choose between '1D' or '2D' or 'HARD_LABEL'
    _C.DATA.TEST_SPLIT_RATIO = 0.2
    _C.DATA.BALANCED_SAMPLER = True
    _C.DATA.OVERSAMPLE_FACTOR = 1.0
    _C.DATA.IGNORE_NOFINDING = False
    _C.DATA.NOFINDING_WEIGHT_GIVEN = False
    # Model settings
    _C.MODEL = CN()
    _C.MODEL.NUM_CLASSES = 15  # 14 pathology classes + 1 background/no finding
    _C.MODEL.BACKBONE = 'resnet50'
    _C.MODEL.PROBABILISTIC = True

    # Training settings
    _C.TRAIN = CN()
    _C.TRAIN.BASE_LR_BBOX = 0.005
    _C.TRAIN.BASE_LR_CLASS = 0.0005
    _C.TRAIN.BASE_LR = 0.001
    _C.TRAIN.BOX_REG_LOSS_WEIGHT = 10.0
    _C.TRAIN.SCHEDULER = 'main_scheduler' # Choose between 'main_scheduler' or 'secondary_scheduler'
    _C.TRAIN.MILESTONES = [10, 20]
    _C.TRAIN.GAMMA = 0.1
    _C.TRAIN.BATCH_SIZE = 4
    _C.TRAIN.OVERFIT_BATCHES = 5
    _C.TRAIN.NUM_EPOCHS = 30
    _C.TRAIN.EARLY_STOPPING = False

    # Augmentation settings
    _C.AUGMENTATION = CN()
    _C.AUGMENTATION.HORIZONTAL_FLIP_PROB = 0.3
    _C.AUGMENTATION.ROTATION_DEGREES = 15
    _C.AUGMENTATION.BRIGHTNESS_RANGE = (0.7, 1.3)
    _C.AUGMENTATION.CONTRAST_RANGE = (0.7, 1.3)
    _C.AUGMENTATION.NOISE_STD = 0.03
    _C.AUGMENTATION.RARE_CLASS_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    _C.AUGMENTATION.RARE_CLASS_AUGMENT_PROB = 1.0
    _C.AUGMENTATION.COMMON_CLASS_AUGMENT_PROB = 0.2

    # Miscellaneous settings
    _C.MISC = CN()
    _C.MISC.RUN_NAME = 'vindr_cxr_softlabel_1d'
    _C.MISC.RESULTS_CSV = os.path.join(project_root_ext, 'outputs/training_metric_logs')
    _C.MISC.SAVE_MODEL_PATH = os.path.join(project_root_ext, 'outputs/saved_models')
    _C.MISC.EVALUATION_RESULTS_DIR = os.path.join(project_root_ext, 'outputs/evaluation_results')
    _C.MISC.PRETRAINED_WEIGHTS = ''
    _C.MISC.FROZEN_LAYERS = []
    _C.MISC.NO_CUDA = False

    return _C.clone()

def _resolve_paths_relative_to_config(cfg, cfg_file):
    """Turn non-absolute path strings in cfg into paths relative to the config file directory."""
    base = os.path.dirname(os.path.abspath(cfg_file))
    def resolve(section_name, key):
        section = getattr(cfg, section_name)
        val = getattr(section, key)
        if not isinstance(val, str) or not val.strip():
            return
        if os.path.isabs(val):
            return
        resolved = os.path.normpath(os.path.join(base, val))
        setattr(section, key, resolved)
    # resolve("DATA", "ROOT")
    resolve("DATA", "SOFT_LABEL_FILE_1D")
    resolve("DATA", "SOFT_LABEL_FILE_2D")
    resolve("DATA", "HARD_LABEL_FILE")
    resolve("MISC", "RESULTS_CSV")
    resolve("MISC", "SAVE_MODEL_PATH")
    resolve("MISC", "EVALUATION_RESULTS_DIR")
    resolve("MISC", "PRETRAINED_WEIGHTS")


def get_cfg_from_file(cfg_file):
    """
    Load configuration from a file.
    Args:
        cfg_file (str): Path to the configuration file.
    Returns:
        CfgNode: Configuration object.
    """
    with open(cfg_file, 'r') as file:
        cfg_dict = yaml.safe_load(file)

    cfg = get_cfg_defaults()
    cfg.merge_from_other_cfg(CN(cfg_dict))
    _resolve_paths_relative_to_config(cfg, cfg_file)
    cfg.freeze()
    return cfg