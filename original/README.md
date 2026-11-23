# Original research code

This folder is the code as I first wrote it, exploratory files included, kept as a record.

What this means for you:

- It is not maintained code. It will not run as-is: the input data files and outputs are not included, and some
  settings (paths, keys) are placeholders.
- Some of this code contains the issues listed in the audit table of the main [README](../README.md) (look-ahead,
  selection bias, missing costs, and so on). The maintained version in the main package fixes them. Read the two side
  by side to see what changed and why.

Contents:

- `.`: main research code (the monthly, forecast-driven allocator)
- [`min_variance/`](min_variance/): the later non-ML follow-up (Ledoit-Wolf minimum variance blended with vol
  parity, daily cadence) and its experiment scripts. It imported modules from the main research repo that are not
  part of this record, so it does not run on its own. The maintained version is
  [`src/volbudget/min_variance/`](../src/volbudget/min_variance/), and
  [docs/min_variance.md](../docs/min_variance.md) lists what I changed and why. Real-data figures in its comments
  and print statements are replaced with `<real figure removed>`, and the result memos that went with it are left
  out because they report real-data numbers.
