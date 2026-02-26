> **Note:** This design doc is historical. The implementation lives in crabpath/*.py. See ARCHITECTURE_REVIEW.md for current architecture.

# CrabPath Parameter Tuning Guide

This file lists the knobs currently covered by runtime autotuning.

## Tunable knobs (5)

`self_tune()` can modify these five values:

| Parameter | Default | Autotune range | What it controls |
|---|---:|---:|---|
| `decay_half_life` (`DecayConfig`) | `80` | `30` - `200` | Edge decay speed between maintenance intervals. |
| `promotion_threshold` (`SynaptogenesisConfig.promotion_threshold`) | `2` | `2` - `6` | Proto-co-firing count needed before edge promotion. |
| `hebbian_increment` (`SynaptogenesisConfig.hebbian_increment`) | `0.06` | `0.02` - `0.12` | Co-fire reinforcement magnitude for real edges. |
| `helpfulness_gate` (`SynaptogenesisConfig.helpfulness_gate`) | `0.1` | `0.05` - `0.5` | Minimum retrieval score required to emit a positive RL signal. |
| `harmful_reward_threshold` (`SynaptogenesisConfig.harmful_reward_threshold`) | `-0.5` | `-1.0` - `-0.2` | Negative score cutoff used for RL harm filtering. |

## Fixed constants (5)

| Parameter | Value | Why fixed |
|---|---:|---|
| `skip_factor` (`SynaptogenesisConfig.skip_factor`) | `0.9` | Constant skip penalty multiplier for candidate routing. |
| `dormant_threshold` (`SynaptogenesisConfig.dormant_threshold`) | `0.1` | Fixed tier boundary for dormant visibility. |
| `reflex_threshold` (`SynaptogenesisConfig.reflex_threshold`) | `0.9` | Fixed tier boundary for reflex (auto-follow) edges. |
| `sibling_weight` (`MitosisConfig.sibling_weight`) | `0.25` | Start low — edges earn weight through co-firing. |
| `sibling_jitter` (`MitosisConfig.sibling_jitter`) | `0.1` | Deterministic ±jitter on sibling init to break uniform softmax deadlock. |
| `temperature` (`LearningConfig.temperature`) | `0.5` | Softmax temperature for PG updates. Lower = sharper gradients from similar weights. |
| `max_outgoing` (`SynaptogenesisConfig.max_outgoing`) | Workspace-derived default | Structural limit; not autotuned. |

`self_tune()` flow: `measure_health()` → `autotune()` → `apply_adjustments()`.

## LLM model rule

**All CrabPath LLM calls use `gpt-5-mini` with `reasoning_effort="minimal"` (0 reasoning tokens).**

Every internal LLM task is classification — routing, scoring, splitting, merging, neurogenesis.
None require chain-of-thought. With default reasoning, gpt-5-mini burns 256+ thinking tokens
on trivial tasks (3× slower, identical output). `reasoning_effort="minimal"` disables this:
1.2s per call, 0 reasoning tokens, same quality. Don't remove this parameter.
