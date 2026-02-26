# User Profile

I am a practical engineer running long-lived agent workflows. I prefer reliable systems that explain themselves.

What I need from tools:

- Clear action labels with explicit file paths.
- Consistent outputs I can parse in automation.
- Deterministic behavior for local reproducibility.
- Minimal surprise from defaults.

Behavioral preferences:

- I like concise responses with direct next steps.
- I expect failures to include machine-readable messages.
- I prefer conservative operations first; if confidence is low, ask for confirmation.
- I want a quick validation signal (like health summary) before trusting a bootstrap.

Work priorities:

1. Keep context quality high and cleanly sourced.
2. Detect low-signal or noisy edges quickly.
3. Preserve the ability to rollback or re-run with fewer assumptions.
4. Keep personal and private data out of persistent outputs.

Decision habits:

- I trust small incremental checks over large one-shot changes.
- I favor explicit configuration over implicit environment dependencies.
- I appreciate concise reasoning and explicit caveats.

Daily working assumptions:

- I prefer stable contexts over maximal contexts.
- I prefer conservative defaults with explicit overrides.
- I prefer deterministic graph snapshots over hidden inference.
- I prefer short summaries with direct references to file paths and actions.

In difficult situations where data is noisy, I rely on validation commands before declaring completion. If a migration changes expected shapes, I pause and compare before continuing.

Decision preferences:

- Choose safer migration parameters first and only expand when observed results are clear.
- Keep output files small and auditable.
- Treat unknown or malformed sessions as skippable rather than blocking the full workflow.
- Keep explanation outputs with seed, suppression, and path details to shorten debugging loops.

Expectation:

I expect each tool command to provide a clean, parseable response with enough context to script checks and alerts. I expect failures to be explicit and not crash the surrounding automation.
