# Query by Sample Distance Comparison

This project provides a script, `query_by_sample_dist_comparison.py`, for comparing sample distances. Follow the instructions below to set up and execute the code.

## Installation

1. Clone the repository or download the files.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
# or, using uv
uv pip install -r requirements.txt
```

## Usage

Run the script from the command line:

```bash
python query_by_sample_dist_comparison.py dataset_path queries_path [params]
```
### Parameters

- `dataset_path`  
    Path to the dataset file. **(Required, positional argument)**

- `queries_path`  
    Path to the queries file. **(Required, positional argument)**

- `-k`, `--top_k`  
    Number of top results to return. Default: `1`.

- `--bins_per_channel`  
    Number of bins per channel for histogram calculation. Default: `32`.

- `--color_space`  
    Color space to use (e.g., `HSV`, `RGB`). Default: `HSV`.

- `--test`  
    Run with a range of bins and color spaces.

- `--output_file`  
    Path to save the results CSV file. Default: `results1.csv`.

### Example

```bash
python query_by_sample_dist_comparison.py ./data/BBDD ./data/qsd1_w1 --top_k 5 --bins_per_channel 16 --color_space RGB --output_file results/test_output.csv --metric cosine
```
