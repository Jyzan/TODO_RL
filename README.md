# TODO_RL_benchmarks
This repository contains BrowseComp，HLE and GAIA benchmarks.:smile:

## Environment
Instead of configuring your environmental variable in your system, we adjust it to .env file configuration, you can change your API or BASE_URL or DEFAULT_MODEL in the .env file!!!

## Quick Start
if you wanna use the multimodal python script, run:
```python
python run_flash_searcher_mm.py --infile mm/hle.jsonl --outfile hle.json --summary_interval <on your need> --concurrency <on your need>
```
**Notitce** that the outfile path doesn't contain the prefix path, because we automaically add it in python script, all you need is just to specify the file name such as `hle.json`

### Update New Feature!
- new Functionality here:

place the result file(json/jsonl) in the output_for_analysis repository and run the `validate.py` module, you should get the **Accuracy** analysis!
