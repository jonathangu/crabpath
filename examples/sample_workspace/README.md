# Sample Workspace

This is a tiny workspace for trying CrabPath. Run:

```bash
crabpath init --workspace examples/sample_workspace --output /tmp/sample-brain
crabpath doctor --state /tmp/sample-brain/state.json
crabpath query "how do I deploy" --state /tmp/sample-brain/state.json --top 3
crabpath inject --state /tmp/sample-brain/state.json \
  --id "fix::1" --content "Never skip CI" --type CORRECTION
crabpath query "skip CI" --state /tmp/sample-brain/state.json --top 3
```
