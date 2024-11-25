#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
from .adopt_optimizer import ADOPT

def Optimizer(parameters, lr, weight_decay, **kwargs):
    """
    Initialize the ADOPT optimizer.
    Only passes valid arguments to the optimizer.
    """
    print("Initialized ADOPT optimizer")

    # Extract valid arguments for ADOPT
    valid_kwargs = {k: v for k, v in kwargs.items() if k in ["decouple", "momentum"]}
    
    return ADOPT(parameters, lr=lr, weight_decay=weight_decay, **valid_kwargs)
