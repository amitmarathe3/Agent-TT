# AGENTS.md

## Python
- Always run Python as `C:\Users\HP\AppData\Local\Programs\Python\Python314\python.exe` (CUDA torch). Never bare `python`.

## Timeouts
- Tool calls time out at about 120s. For anything longer (pytest, training, bootstrap, inference over a split) run `start /b cmd /c "<cmd> > outputs\logs\<name>.log 2>&1"`, poll the log every 60s printing its last 5 lines, and claim a result only after its output file exists on disk. This replaces the old "no background training" rule.

## Replies
- Keep replies short: raw output plus a facts table, no explanations.

## Audit
- After every prompt, run `scripts/audit_project.py` and paste its table. Do not claim a step is done if the audit has a FAIL.
- STOP AND ASK (print the question and end the turn, wait for an answer) when: any audit FAIL; any result looks too good (>0.95) or val/test gap >0.15; a step will take over 30 min; there is a choice between approaches (give 2-3 options with your recommendation and the cost of each); you are about to delete or overwrite anything outside `outputs/_invalid`; a previous claim turns out to be wrong.
- When you must stop, say: what happened, the evidence (raw output), the options, your recommendation.
