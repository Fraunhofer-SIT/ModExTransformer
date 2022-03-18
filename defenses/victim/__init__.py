""" 
    MIT License
    Copyright (c) 2020 Tribhuvanesh Orekondy
    Taken and modified from: https://github.com/tribhuvanesh/prediction-poisoning/blob/master/defenses/victim/__init__.py 

    Apache 2.0 License 
    Copyright (c) 2022, Fraunhofer e.V.
    All rights reserved.   
""""

from .blackbox import Blackbox

#### Cleaned up models
## Blackbox
from .bb_mad import MAD
from .bb_reversesigmoid import ReverseSigmoid   # Reverse Sigmoid noise
from .bb_randnoise import RandomNoise  # Random noise in logit space
