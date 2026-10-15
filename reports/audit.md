# Project Audit

| Check | Status | Detail |
|---|---|---|
| 1 label files & per-class vs data_index | PASS | counts match expected |
| 2 patient split leakage | PASS | no patient in two splits; run_config train disjoint |
| 2b exp_ run completeness | PASS | 1 exp_ run(s) complete |
| 3 leaderboard artifacts & values | PASS | all rows backed by metrics.json |
| 4 metrics.json n_gt guard | PASS | TP+FN == n_gt == label box count |
| 5 doc numbers in metrics/data_index | WARN | 27 unmatched (full list in reports/audit.md) |
| 5 version claims vs git tags | WARN | claims without tag: ['v0.3.0', 'v0.3.1', 'v0.3.2'] |
| 6 suspicious results | PASS | no metric >0.95; no val/test gap >0.15 |
| 7 environment | PASS | python=3.14.8; torch=2.14.1+cu126; cuda.is_available=True; git=dirty (9 files); tags=['v0.1.0', 'v0.2.0', 'v0.2.1'] |
| 8 frame-classifier runs | PASS | NOT PRESENT (no frame-classifier runs) |

SUMMARY: 10 checks | FAIL=0 WARN=2

## Unmatched README/CHANGELOG decimals (3+ places)

README.md:294: 0.5066
README.md:294: 0.5066
README.md:294: 0.5066
README.md:295: 0.1385
README.md:295: 0.1385
README.md:295: 0.1385
README.md:344: 0.6363
README.md:513: 0.4003
README.md:513: 0.8034
README.md:520: 0.3815
README.md:520: 0.7799
README.md:534: 0.001
README.md:572: 0.001
README.md:600: 0.0749
README.md:649: 0.8124
README.md:649: 0.7415
README.md:649: 0.6510
README.md:649: 0.6842
README.md:649: 0.6390
README.md:649: 0.6890
README.md:650: 0.6715
README.md:650: 0.0218
README.md:650: 0.9240
README.md:650: 0.0165
README.md:655: 0.6004
CHANGELOG.md:26: 0.001
CHANGELOG.md:52: 0.6004