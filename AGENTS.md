# AGENTS.md

## Python
- Always run Python as `C:\Users\HP\AppData\Local\Programs\Python\Python314\python.exe` (CUDA torch). Never bare `python`.

## Timeouts
- Tool calls time out at about 120s. For anything longer (pytest, training, bootstrap, inference over a split) run `start /b cmd /c "<cmd> > outputs\logs\<name>.log 2>&1"`, poll the log every 60s printing its last 5 lines, and claim a result only after its output file exists on disk. This replaces the old "no background training" rule.

## Replies
- Keep replies short: raw output plus a facts table, no explanations.
