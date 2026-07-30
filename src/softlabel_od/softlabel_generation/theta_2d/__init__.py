"""
theta_2d — canonical 4x4 Sigma_theta (over x, y, w, h) covariance
representation for 2D soft labels.

Public entry points (run as modules, e.g.
    python -m softlabel_od.softlabel_generation.theta_2d.calculate_priors_theta
    python -m softlabel_od.softlabel_generation.theta_2d.gmm_theta
):
    calculate_priors_theta.py  -- step 4 equivalent (class priors)
    gmm_theta.py                -- step 5 equivalent (soft label generation)

Supporting modules:
    vertex_transform.py         -- the proved, verified theta <-> vertex transform
    prior_theta_calculator.py   -- PriorThetaCalculator (replaces Prior2DCalculator)

This package is deliberately kept separate from the existing
softlabel_generation.core.label_assembly.priors / label_constructor
modules -- nothing there is modified. See the project specification
document for the full mathematical justification.
"""
