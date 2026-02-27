# Cold Start Guide

This guide is for a brand new workspace with zero sessions and zero corrections.

## Prerequisites

- Use the repo sample workspace: `examples/sample_workspace`
- Ensure `openclawbrain` is installed (`pip install openclawbrain`)

## 1) Build a fresh brain

```bash
openclawbrain init --workspace examples/sample_workspace --output /tmp/cold-start-brain
openclawbrain doctor --state /tmp/cold-start-brain/state.json
```

Expected:

```text
python_version: PASS
state_file_exists: PASS
state_json_valid: PASS
Summary: 7/7 checks passed
```

## 2) Query before teaching

```bash
openclawbrain query "how do I deploy" --state /tmp/cold-start-brain/state.json --top 3
```

Expected:

```text
deploy.md::0
~~~~~~~~~
How to perform a deployment safely with checklist gates

monitoring.md::0
~~~~~~~~~~~~~
What to watch after a deployment

deploy.md::1
~~~~~~~~~
How to rollback a failed deployment
```

## 3) Mark one outcome as bad, then teach a correction

```bash
openclawbrain learn --state /tmp/cold-start-brain/state.json --outcome -1.0 --fired-ids "deploy.md::0,deploy.md::1"
openclawbrain inject --state /tmp/cold-start-brain/state.json --id fix::hotfix-cicd --content "Never skip CI for hotfixes" --type CORRECTION
```

Expected: learn summary

```json
{"edges_updated":2,"max_weight_delta":0.09}
```

## 3b) Add new knowledge

```bash
openclawbrain inject --state /tmp/cold-start-brain/state.json \
  --id "teaching::deploy-tip" \
  --content "Always notify #ops before deploying on Fridays" \
  --type TEACHING --json
```

Expected: the node connects to deploy.md chunks without creating inhibitory edges.

## 4) Query after routing change

```bash
openclawbrain query "can I skip CI" --state /tmp/cold-start-brain/state.json --top 3
```

Expected: hotfix path stays high and the new correction node appears.

```text
fix::hotfix-cicd
~~~~~~~~~~~~~~~
Never skip CI for hotfixes

onboarding.md::1
~~~~~~~~~~~~~
First week workflow
```

## 5) Ask a third canned question to validate cross-file routing

```bash
openclawbrain query "can I monitor on-call response" --state /tmp/cold-start-brain/state.json --top 3
```

Expected:

```text
incidents.md::0
~~~~~~~~~~~
Before an incident

monitoring.md::0
~~~~~~~~~~~~~
What to watch after a deployment
```

## 6) Route sanity check with health

```bash
openclawbrain health --state /tmp/cold-start-brain/state.json
openclawbrain learn --state /tmp/cold-start-brain/state.json --outcome -1.0 --fired-ids "deploy.md::1,deploy.md::2"
openclawbrain health --state /tmp/cold-start-brain/state.json
```

Look for a health readout that changes after the bad outcome (edges and percentages will shift).
