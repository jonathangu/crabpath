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
| `sibling_weight` (`MitosisConfig.sibling_weight`) | Workspace-derived default | Warm-start behavior only; not autotuned. |
| `max_outgoing` (`SynaptogenesisConfig.max_outgoing`) | Workspace-derived default | Structural limit; not autotuned. |

`self_tune()` flow: `measure_health()` → `autotune()` → `apply_adjustments()`.

## LLM model rule

**All CrabPath LLM calls must use a non-reasoning model (e.g. `gpt-4o-mini`).**

Every internal LLM task is classification — routing, scoring, splitting, merging, neurogenesis.
None require chain-of-thought. Reasoning models (gpt-5-mini, o-series) waste tokens on
internal thinking that produces identical outputs 3× slower. The whole architecture assumes
"one cheap, fast LLM." Don't upgrade the helper model — it makes things worse.
