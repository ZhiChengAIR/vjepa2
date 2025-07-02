import numpy as np
import torch.nn.functional as F

from notebooks.utils.mpc_utils import cem, compute_new_pose


class WorldModel(object):
    pass