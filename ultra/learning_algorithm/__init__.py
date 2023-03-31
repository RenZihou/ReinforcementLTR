# note:
from __future__ import absolute_import
from .base_algorithm import *
from .dla import *
from .ipw_rank import *
from .regression_EM import *
from .pdgd import *
from .dbgd import *
from .pairwise_debias import *
from .naive_algorithm import *
from .mgd import *
from .nsgd import *
from .lambda_rank import *
from .prs_rank import *
from .pg_rank import *
from .reinforce import *
from .ddpg import *


def list_available() -> list:
    from .base_algorithm import BaseAlgorithm
    from ultra.utils.sys_tools import list_recursive_concrete_subclasses
    return list_recursive_concrete_subclasses(BaseAlgorithm)
