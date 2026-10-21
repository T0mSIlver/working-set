# data/

The real-workload scripts (`real_capacity.py`, `real_mns.py`, `warm_whisker.py`)
read every `prompt_tokens*.csv` in this directory, in name order, as one set
of observed prompt lengths each (labelled set A, set B, … in the figures).

Each file is one row per response, with a `prompt_tokens` integer column
(the prompt length in tokens). Prompts shorter than `MIN_TOKENS` (1000) are
dropped as junk during cleaning.

These CSVs contain personal usage data and are **not** committed to the repo
(see `.gitignore`). Drop your own copies here, or point the scripts at another
location with the `DATA_DIR` environment variable:

```bash
DATA_DIR=/path/to/csvs python scripts/real_mns.py
```

`warm_capacity.py` is fully synthetic (it sweeps hypothetical distributions in
its `HYPOTHESES` block) and needs no CSVs.
