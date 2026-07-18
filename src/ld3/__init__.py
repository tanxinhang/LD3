"""LD3: physics-guided TF-DD validation utilities for OFDM-ISAC."""

from .config import OFDMConfig, ChannelConfig
from .channel import PathSet, generate_path_set, synthesize_tf_channel
from .oracle import (
    corrupt_tokens,
    perturb_tokens,
    oracle_path_tokens,
    oracle_path_tokens_v2,
    estimated_path_tokens_v2,
    compute_path_quality,
    oracle_perfect_reconstruction,
    oracle_support_ls_reconstruction,
    estimated_support_ls_reconstruction,
)
from .pilots import make_pilot_mask, observe_pilots, generate_noise_grid

__all__ = [
    "OFDMConfig",
    "ChannelConfig",
    "PathSet",
    "generate_path_set",
    "synthesize_tf_channel",
    "corrupt_tokens",
    "perturb_tokens",
    "oracle_path_tokens",
    "oracle_path_tokens_v2",
    "estimated_path_tokens_v2",
    "compute_path_quality",
    "oracle_perfect_reconstruction",
    "oracle_support_ls_reconstruction",
    "estimated_support_ls_reconstruction",
    "make_pilot_mask",
    "observe_pilots",
    "generate_noise_grid",
]
