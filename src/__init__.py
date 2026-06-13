"""SQAI 2026: Neural Error Mitigation for Variational Quantum Algorithms.

A deep learning framework for mitigating errors in variational quantum algorithms
running on NISQ devices.

Main components:
- quantum: Quantum circuit implementations (VQE, QAOA) and noise models
- models: Neural network architectures for error prediction and mitigation
- training: Data generation and training pipelines
- classical: Classical baseline methods (ZNE, PEC, DD)
- hybrid: Combined neural-classical mitigation pipelines
- utils: Metrics and visualization utilities
"""

__version__ = "0.1.0"
__author__ = "SQAI 2026 Research Team"

from . import quantum
from . import models
from . import training
from . import classical
from . import hybrid
from . import utils
