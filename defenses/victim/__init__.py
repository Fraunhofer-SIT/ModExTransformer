""" Taken and modifided from: 
    Tribhuvanesh Orekondy, https://github.com/tribhuvanesh/prediction-poisoning/blob/master/defenses/victim/__init__.py """"

from .blackbox import Blackbox

#### Cleaned up models
## Blackbox
from .bb_mad import MAD
from .bb_reversesigmoid import ReverseSigmoid   # Reverse Sigmoid noise
from .bb_randnoise import RandomNoise  # Random noise in logit space
