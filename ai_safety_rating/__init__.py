"""Probability-based AI safety rating engine (ENGL 415 research project, Othniel Maganga)."""
from .engine import (  # noqa: F401
    Settings, CategoryRating, ModelLabel, DEFAULT_TIERS, TIER_MEANING,
    beta_cdf, beta_ppf, chance_of_any_unsafe, tier_for,
    rate_category, rate_model, rate_file_data,
)

__version__ = "1.0.0"
