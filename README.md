# VQ-VAE Implementation for FashionMNIST

PyTorch implementation of Vector Quantized Variational Autoencoder (VQ-VAE) for FashionMNIST dataset analysis.

## Model Architecture

- **Encoder**: 28x28x1 → 12x12xembedding_dim
  - Residual blocks and downsampling
  - Batch normalization and ReLU activation

- **Vector Quantizer**: Discrete latent representation
  - Codebook: num_embeddings x embedding_dim
  - Straight-through gradient estimator

- **Decoder**: 12x12xembedding_dim → 28x28x1
  - Upsampling and reconstruction  - 
  - Batch normalization and ReLU activation
 
The figure bellow illustrate this implementation
![](https://github.com/enzolvd/VQ-VAE-fashion_MNIST/blob/main/VQ_VAE_flowchart.png)
## Implementation

### Files Structure
- `model.py`: VQ-VAE architecture (Encoder, Decoder, Vector Quantizer)
- `utils.py`: Training utilities and visualization
- `VQ_VAE.py`: Training script and hyperparameter search
- `results_analysis.ipynb`: Results analysis and latent space visualization

### Hyperparameters
- Number of embeddings: [25, 50, 100]
- Embedding dimensions: [16, 32, 64]
- Training epochs: 50
- Beta (commitment loss): 0.25
- Optimizer: Adam

## Requirements

```
torch
torchvision
tqdm
matplotlib
numpy
pandas
scikit-learn
seaborn
```

## Usage

1. Train model:
```bash
python VQ_VAE.py
```

2. Visualize training results:
```python
from utils import load_losses, plot_losses, find_best_model_params

# Load and visualize training results
losses = load_losses('./checkpoints/losses_dicts')
plot_losses(losses)
best_params = find_best_model_params(losses)
```

3. Run analysis notebook for detailed visualizations and insights

## References:
[^1]: van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2018). Neural Discrete Representation Learning. arXiv preprint arXiv:1711.00937.
