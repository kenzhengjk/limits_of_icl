# In-Context Learning with Transformers

This repository contains code for training and evaluating transformer models on in-context learning tasks, with support for various attention mechanisms and curriculum learning.

## Setup

### Requirements

Install dependencies using the provided requirements file:

```bash
pip install -r requirements.txt
```

**Note**: This project requires PyTorch with CUDA support. The requirements include `torch==2.9.1+cu130`.

## Project Structure

```
.
├── train.py              # Main training script
├── eval.py               # Evaluation utilities
├── eval_colab.ipynb      # Colab notebook for evaluation
├── wrapper_model.py      # Model wrapper classes
├── nanogpt_model.py      # NanoGPT implementation
├── attentions.py         # Attention mechanism implementations
├── tasks.py              # Task definitions (linear regression, etc.)
├── samplers.py           # Data samplers
├── curriculum.py         # Curriculum learning logic
├── train_utils.py        # Training utilities
├── plot_loss.py          # Loss curve plotting
├── plot_utils.py         # Evaluation plotting utilities
└── requirements.txt      # Python dependencies
```

## Training

### Configuration

Create a YAML configuration file with the following structure:

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

wandb:                               # Optional W&B logging
  entity: your-entity
  project: your-project
  name: experiment-name
  log_every_steps: 100

test_run: false                      # Set true for quick testing
```

### Supported Attention Types

- `softmax_causal`: Standard causal softmax attention
- `local`: Local attention with windowing
- `mqa`: Multi-query attention
- `rela`: ReLA attention
- `favor`: FAVOR+ linear attention
- `rebased`: ReBased attention

### Supported Tasks

- `linear_regression`: Standard linear regression
- `sparse_linear_regression`: Sparse linear regression with k active dimensions
- `noisy_linear_regression`: Linear regression with Gaussian noise
- `affine_regression`: Linear regression with bias term
- `uniform_linear_regression`: Linear regression with uniform weight distribution
- `quadratic_regression`: Quadratic regression
- `relu_2nn_regression`: Two-layer ReLU neural network regression
- `decision_tree`: Decision tree approximation
- `linear_classification`: Binary linear classification

### Run Training

```bash
python train.py --config configs/my_config.yaml
```

The training script will:
1. Create the output directory specified in `out_dir`
2. Train the model with curriculum learning
3. Save checkpoints periodically to `out_dir/state.pt`
4. Log training progress to `out_dir/training_log.csv`
5. Generate a loss curve plot at the end

### Training Outputs

- `state.pt`: Latest model checkpoint (includes optimizer state)
- `model_{step}.pt`: Model checkpoints at specified intervals
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
3. Update the paths in the notebook to point to your model directories
4. Run the evaluation cells to generate metrics and plots

### Evaluation Metrics

The evaluation suite compares trained transformers against baselines:
- Least Squares
- k-Nearest Neighbors
- Lasso Regression (for sparse tasks)
- XGBoost (for decision trees)
- Greedy Tree Learning (for decision trees)
- 2-layer Neural Network with Gradient Descent

Metrics include:
- Mean squared error across in-context examples
- Bootstrap confidence intervals
- Performance vs. context length curves

## Plotting

### Loss Curves

Generate loss curves from training logs:

```python
from plot_loss import plot_full_loss_curve

plot_full_loss_curve(
    BASE_DIR="./models/my_experiment",
    ATTENTION_TYPE="softmax_causal",
    CSV_NAME="training_log.csv",
    window_size=15,
    dims_int=2000  # Curriculum step interval
)
```

### Evaluation Plots

Use utilities from `plot_utils.py`:

```python
from plot_utils import basic_plot, plot_param_sweep, plot_experiment

# Basic comparison plot
fig, ax = basic_plot(metrics, models=["Softmax", "FAVOR", "Local"])

# Parameter sweep visualization
figs = plot_param_sweep(all_metrics, models_to_plot=["Softmax", "FAVOR"])

# Single experiment plot
fig, ax = plot_experiment(all_metrics, "standard", models_to_plot=["Softmax"])
```

## Example Configurations

### Minimal Configuration

```yaml
model:
  family: nanogpt
  n_dims: 20
  n_embd: 128
  n_head: 4
  n_layer: 6
  n_positions: 41

out_dir: ./models/test_run

training:
  train_steps: 10000
  task: linear_regression
  data: gaussian

test_run: true
```

### Sparse Linear Regression

```yaml
model:
  family: nanogpt
  n_dims: 20
  n_embd: 256
  n_head: 8
  n_layer: 12
  n_positions: 41

training:
  task: sparse_linear_regression
  task_kwargs:
    sparsity: 3  # Only 3 active dimensions
```

### Local Attention

```yaml
model:
  attention_type: local
  attention_kwargs:
    window_size: 8  # Attention window size
```

## Citation

If you use this code, please cite the relevant papers on in-context learning and attention mechanisms.
