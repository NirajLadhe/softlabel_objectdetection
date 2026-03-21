"""
PyTorch Lightning adapters (experimental — parallel to vanilla PyTorch path).

These modules mirror the vanilla training logic and will become the primary
training path once the migration is complete.

1D GMM
------
VinDr1DLightningModule   — LightningModule for 1D soft label training
VinDr1DDataModule        — LightningDataModule for VinDr-CXR 1D

Hard Label
----------
VinDrHardLightningModule — LightningModule for hard label training
VinDrHardDataModule      — LightningDataModule for VinDr-CXR hard labels
"""
