# Project Audit

| Check | Status | Detail |
|---|---|---|
| 1 label files & per-class vs data_index | PASS | counts match expected |
| 2 patient split leakage | PASS | no patient in two splits; run_config train disjoint |
| 2b exp_ run completeness | PASS | 5 exp_ run(s) complete |
| 3 leaderboard artifacts & values | PASS | all rows backed by metrics.json |
| 4 metrics.json n_gt guard | PASS | TP+FN == n_gt == label box count |
| 5 doc numbers in metrics/data_index | PASS | all 3+ decimal numbers found |
| 5 version claims vs git tags | PASS | 3 claims all tagged |
| 6 suspicious results | PASS | no metric >0.95; no val/test gap >0.15 |
| 7 environment | PASS | python=3.14.8; torch=2.14.1+cu126; cuda.is_available=True; git=dirty (7 files); tags=['v0.1.0', 'v0.2.0', 'v0.2.1'] |
| 8 frame-classifier runs | PASS | 4 eval dir(s); zero patient overlap between splits |

SUMMARY: 10 checks | FAIL=0 WARN=0