"""vm_micro.dl  Deep-learning depth prediction models.

Provides the full training + inference pipeline for both classification
and regression tasks on airborne (FLAC) and structure-borne (HDF5) audio.
"""

from .config import TrainConfig
from .models import DepthModel
from .training import fit_final_model_all_files, fit_repeated_experiment

__all__ = ["TrainConfig", "DepthModel", "fit_repeated_experiment", "fit_final_model_all_files"]
