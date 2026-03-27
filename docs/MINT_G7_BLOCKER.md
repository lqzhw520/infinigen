# G7 Blocker: LeRobotDataset pyarrow crash corrupts dataset

## Issue
LeRobotDataset.create() + add_frame() + save_episode() succeeds, but Python crashes on exit
due to pyarrow arrow cleanup error:
```
Check failed: _s.ok() Operation failed: internal::ImportDecimalType(&decimal_type_)
Bad status: Unknown error: sys.meta_path is None, Python is likely shutting down.
```

This crash corrupts the dataset files, specifically:
- `meta/episodes/` directory is never created
- `meta/info.json` may be incomplete

## Impact
- G7 (MINT batch load) fails — cannot load dataset
- G8 (MINT train) will also fail — needs valid dataset
- G9 (eval) cannot proceed without G8

## Root cause
pyarrow's Python shutdown cleanup conflicts with LeRobot's internal state.
This is a known issue with pyarrow on Python 3.12 in certain environments.

## Workaround options
1. Use Python 3.11 instead of 3.12 for the mint env
2. Patch LeRobotDataset to not use pyarrow for episodes metadata
3. Create dataset in a subprocess and let it crash without affecting parent
4. Use datasets.Dataset.to_parquet() directly instead of LeRobotDataset

## Status
- G1-G6: PASS
- G7-G8: BLOCKED by this issue
- G9: Depends on G8

## Recommended fix
Switch mint env to Python 3.11 or patch pyarrow shutdown behavior.
