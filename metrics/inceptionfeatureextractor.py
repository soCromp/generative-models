# https://github.com/devavratTomar/torch-fidelity/blob/37d34e7d64cb493e5fe10be99b1ff1a4ab1a3f64/torch_fidelity/feature_extractor_inceptionv3.py

import sys
from contextlib import redirect_stdout

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from torch_fidelity.feature_extractor_base import FeatureExtractorBase
from torch_fidelity.helpers import vassert
from torch_fidelity.interpolate_compat_tensorflow import interpolate_bilinear_2d_like_tensorflow1x

