# debug_backbone.py — run this standalone first
import timm
import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

print("=== Step 1: Load Lunit model ===")
lunit = timm.create_model('hf-hub:1aurent/resnet50.lunit_swav', pretrained=True)
lunit_state = lunit.state_dict()
print(f"Keys: {len(lunit_state)}")
print(f"First 5: {list(lunit_state.keys())[:5]}")

print("\n=== Step 2: Create FPN backbone ===")
backbone = resnet_fpn_backbone('resnet50', weights=None, trainable_layers=3)
backbone_state = backbone.state_dict()
print(f"First 5 FPN keys: {list(backbone_state.keys())[:5]}")

print("\n=== Step 3: Map and load ===")
mapped = {
    f"body.{k}": v
    for k, v in lunit_state.items()
    if f"body.{k}" in backbone_state
    and backbone_state[f"body.{k}"].shape == v.shape
}
print(f"Mapped: {len(mapped)} keys")
print(f"NOT mapped from Lunit: {[k for k in lunit_state if f'body.{k}' not in mapped]}")
backbone.load_state_dict(mapped, strict=False)

# Verify weights actually changed
before_zeros = torch.zeros_like(backbone_state['body.conv1.weight'])
after = backbone.state_dict()['body.conv1.weight']
print(f"conv1 changed from zero: {not torch.allclose(after, before_zeros)}")
print(f"conv1 sample: {after[0,0,0,:4].tolist()}")
assert len(mapped) > 100, f"FAIL: only {len(mapped)} keys mapped — check key names"

print("\n=== Step 4: Normalization ===")
cfg = timm.data.resolve_model_data_config(lunit)
print(f"Lunit mean: {list(cfg['mean'])}")
print(f"Lunit std:  {list(cfg['std'])}")
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]
mean_ok = all(abs(a-b)<0.001 for a,b in zip(cfg['mean'], imagenet_mean))
std_ok  = all(abs(a-b)<0.001 for a,b in zip(cfg['std'], imagenet_std))
print(f"Matches ImageNet: mean={mean_ok}, std={std_ok}")
if not mean_ok or not std_ok:
    print("ACTION NEEDED: pass these to FasterRCNN constructor:")
    print(f"  image_mean={list(cfg['mean'])}, image_std={list(cfg['std'])}")
else:
    print("No normalization change needed in FasterRCNN")

print("\nDEBUG COMPLETE")