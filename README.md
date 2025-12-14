# Not All Attention Is Created Equal: Scaling ICL Robustness Across Mechanisms

By Joshua Lu, Ken Zheng, Lenci Ni, Tiger Zhang, William Li 
---

This repository contains code for training and evaluating transformer models on in-context learning tasks, with support for various attention mechanisms and curriculum learning.

## Setup

### Requirements
Create a conda environment using `requirements.txt`:

```bash
conda create --name limits_of_icl --file requirements.txt
```

Then activate:

```bash
conda activate limits_of_icl
```

If any dependencies are not installed automatically, you can install them using:

```bash
pip install -r requirements.txt
```
**Note**: This project requires PyTorch with CUDA support.

## Training

### Supported Attention Types

- `softmax_causal`: Standard causal softmax attention
- `local`: Local attention with windowing
- `mqa`: Multi-query attention
- `rela`: ReLA attention
- `favor`: FAVOR+ linear attention
- `rebased`: ReBased attention

### Run Training
To run training on a dummy file, you can use the `src/dummy.yaml` config file.

```bash
cd src
python train.py --config configs/dummy.yaml
```

### Training Outputs

- `state.pt`: Latest model checkpoint (includes optimizer state)
- `training_log.csv`: Training metrics (step, loss, n_points, n_dims)
- `{attention_type}_loss.jpg`: Loss curve visualization

## Evaluation

### Using the Colab Notebook

1. Open `eval_colab.ipynb` in Google Colab
2. Mount your Google Drive containing trained models:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Create a YAML configuration file with the following structure for the model you want to evaluate. Place this file under the corresponding model's directory:

```yaml
model:
  family: nanogpt                    # Model family: 'nanogpt' or 'gpt2'
  n_dims: 20                         # Input dimensionality
  n_embd: 256                        # Embedding dimension
  n_head: 8                          # Number of attention heads
  n_layer: 12                        # Number of transformer layers
  n_positions: 41                    # Maximum sequence length
  attention_type: softmax_causal     # Attention type (see below)
  attention_kwargs: {}               # Additional attention parameters

out_dir: ./models/my_experiment      # Output directory for checkpoints

training:
  batch_size: 64
  learning_rate: 0.0001
  train_steps: 100001
  save_every_steps: 1000             # Checkpoint frequency
  keep_every_steps: 100000           # Keep checkpoint at these intervals
  
  task: linear_regression            # Task type (see tasks.py)
  task_kwargs: {}                    # Task-specific parameters
  data: gaussian                     # Data sampler type
  
  num_tasks: null                    # Number of tasks (null = unlimited)
  num_training_examples: null        # Training examples (null = unlimited)
  
  curriculum:                        # Curriculum learning schedule
    dims:
      start: 5                       # Starting dimensionality
      end: 20                        # Final dimensionality
      inc: 1                         # Increment per step
      interval: 2000                 # Steps between increments
    points:
      start: 11                      # Starting context length
      end: 41                        # Final context length
      inc: 2                         # Increment per step
      interval: 2000                 # Steps between increments

test_run: false                      # Set true for quick testing
```
3. Update the paths in the notebook to point to your model directories
4. Run the evaluation cells to reproduce our results and generate results of your own!
