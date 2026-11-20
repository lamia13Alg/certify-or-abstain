"""scrc — Shift-aware Conformal Risk Control for 3D medical image segmentation.

Modules
-------
summary   : compress a 3D probability map into a patient-level summary (histograms)
losses    : patient-level loss curves L_i(lambda) computed from the summaries
crc       : conformal risk control (standard, weighted / covariate-shift, truncated)
weights   : density-ratio estimation in embedding space (cross-fitted domain classifier)
selective : patient-level abstention (joint-risk CRC and Learn-then-Test)
volume    : conformal intervals on organ volume (standard and weighted, CQR)
synthetic : synthetic covariate-shift generator used to check the guarantees
"""

__version__ = "0.1.0"
