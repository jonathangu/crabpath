# Upgrading OpenClawBrain

## Quick upgrade (same version, new fixes)

```bash
launchctl bootout gui/501 ~/Library/LaunchAgents/com.openclawbrain.<brain>.plist
cd ~/openclawbrain && git pull
pip install -e '.[openai]'
launchctl bootstrap gui/501 ~/Library/LaunchAgents/com.openclawbrain.<brain>.plist
```

## Full rebuild (new major version)

```bash
launchctl bootout gui/501 ~/Library/LaunchAgents/com.openclawbrain.<brain>.plist
cd ~/openclawbrain && git pull
pip install -e '.[openai]'
openclawbrain init --workspace <workspace> --output <brain-dir> --embedder openai
openclawbrain replay --state <brain-dir>/state.json --sessions <session-dir>
openclawbrain maintain --state <brain-dir>/state.json --tasks health,decay,scale,split,merge,prune,connect --embedder openai
launchctl bootstrap gui/501 ~/Library/LaunchAgents/com.openclawbrain.<brain>.plist
```

## Notes

NOTE: On macOS, crontab writes are blocked from non-TTY processes
by TCC (Transparency, Consent, and Control). If running from an
agent framework, run crontab commands manually from Terminal.app.
LaunchAgents are the recommended alternative to crontab on macOS.
