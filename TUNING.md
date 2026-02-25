# CrabPath Parameter Tuning Guide

This file lists all parameters that `self_tune()` can modify through the autotuner.

## Synaptogenesis/autotune parameters

| Parameter | Default | Autotune range | What it controls |
|---|---:|---:|---|
| `decay_half_life` (from `DecayConfig`) | `80` | `30` - `200` | Exponential decay speed of edge weights between maintenance intervals. Lower values decay faster and tighten routing spread. |
| `promotion_threshold` (`SynaptogenesisConfig.promotion_threshold`) | `2` | `2` - `6` | How many proto-co-firing credits are required before edge promotion. Higher values reduce noisy edge creation. |
| `hebbian_increment` (`SynaptogenesisConfig.hebbian_increment`) | `0.06` | `0.02` - `0.12` | Per co-fire reinforcement applied to existing edges. Higher values move edges up tiers faster. |
| `skip_factor` (`SynaptogenesisConfig.skip_factor`) | `0.9` | `0.80` - `0.98` | Weight multiplier for skipped candidates. Lower values punish weak candidates more. |
| `reflex_threshold` (`SynaptogenesisConfig.reflex_threshold`) | `0.8` | `0.70` - `0.95` | Minimum edge weight to mark a route as reflex (auto-follow). |
| `dormant_threshold` (`SynaptogenesisConfig.dormant_threshold`) | `0.3` | `0.15` - `0.45` | Tier boundary for dormant edges. Higher values hide weaker edges from router visibility. |
| `helpfulness_gate` (`SynaptogenesisConfig.helpfulness_gate`) | `0.1` | `0.05` - `0.5` | Minimum retrieval score required to emit a positive RL signal. Higher values suppress weak positives. |
| `harmful_reward_threshold` (`SynaptogenesisConfig.harmful_reward_threshold`) | `-0.5` | `-1.0` - `-0.2` | Minimum node score considered harmful. More negative values require stronger negative evidence before punishing. |

## Autotune control flow

- `self_tune()` computes `GraphHealth`, calls `autotune()`, then applies `apply_adjustments()`.
- Adjustable knobs are limited to the parameters above.
- Cooldown and max-adjustment caps in `SafetyBounds` prevent rapid oscillation.
