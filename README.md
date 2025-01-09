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
  - Mirrored encoder architecture
  - Upsampling and reconstruction

[![](https://mermaid.ink/img/pako:eNqdVe1umzAUfRXk_iVdMIEAmvpjzSZV6rS10_ZjpKoMNokVwNSYNG3V59gD7cXmD6CEfq4osS7Xx8fX956L70DKMAERyHJ2na4RF9bp-bK05FM3yYqjam2dlFUjjE89mHKSCsrKHqkeqkCxhlonBVoRCwbWzgzOhcGREi_LEfnnUgXAX6NPWbl14i-U18I6lrb1KWfpxvqY8A9Hg5382cXDGk5qJz4n9RugmF2XTryQY42KKict2pmp6PUAg31m2DO_DFXMcMDcoqEGyoEUCcGYlqtLTIv9Ldz4OyeTqwaVwhpt9uLyp9L86-z1DGOSMLaJj1ujzUGZXG4lnPH675_ngtUxxmdqpLdIUb8QyoLoiltPBXR-OlQUVYKigqLc1Hx8-EdFmY3L_Ry2qZz4Z_VMrcci8h6xPg1tKtiTdmrbyZ8L9_n8kXQeNLmHZLKfvjVCNZQ6vca-rafMqtfyyzSq2-INPatb3JpMjkw3Gqc2tVO1m_EpS7t0WxmfNjsc7HGwx8EHHOxwrvH1aFdPaLV1u7daHfnNaFpHTSktdYegoqOf9cQz7ZKqMB5pdBivx3gtBnaYPky_x_jaxVTyjY-1GWNtSdqwc1TXC5JZmGSoyYWV0TyPDrIws2vB2YZEB67rtvbkmmKxjmC1s1OWMx4dTKfTEQ8xX9GOJ0v81Hkn1faqZQmDLAmDd7JgMgwomKfyoP9HNSBsldcmazih1WcrvdlaYMqE2oS21gsZXi_dGqMZ28hje7W_ExVq5cyWGlCGJw3N5dtMx6D5gA0KwgtEsbw67xTBEog1KcgSRNLEiG-WYFneSxxqBPtxU6YgErwhNuCsWa1BlKG8lm9NhZEgC4pk5xYdpELlb8aKHiTfQXQHdiDy54ehG0w9P5x5wXQ6D2xwAyIYwkP5n3ueA_3Qc517G9xqAucQQpnRwHHCuSuXhI4NCKbya_7VXPv69r__B73EbBI?type=png)](https://mermaid.live/edit#pako:eNqdVe1umzAUfRXk_iVdMIEAmvpjzSZV6rS10_ZjpKoMNokVwNSYNG3V59gD7cXmD6CEfq4osS7Xx8fX956L70DKMAERyHJ2na4RF9bp-bK05FM3yYqjam2dlFUjjE89mHKSCsrKHqkeqkCxhlonBVoRCwbWzgzOhcGREi_LEfnnUgXAX6NPWbl14i-U18I6lrb1KWfpxvqY8A9Hg5382cXDGk5qJz4n9RugmF2XTryQY42KKict2pmp6PUAg31m2DO_DFXMcMDcoqEGyoEUCcGYlqtLTIv9Ldz4OyeTqwaVwhpt9uLyp9L86-z1DGOSMLaJj1ujzUGZXG4lnPH675_ngtUxxmdqpLdIUb8QyoLoiltPBXR-OlQUVYKigqLc1Hx8-EdFmY3L_Ry2qZz4Z_VMrcci8h6xPg1tKtiTdmrbyZ8L9_n8kXQeNLmHZLKfvjVCNZQ6vca-rafMqtfyyzSq2-INPatb3JpMjkw3Gqc2tVO1m_EpS7t0WxmfNjsc7HGwx8EHHOxwrvH1aFdPaLV1u7daHfnNaFpHTSktdYegoqOf9cQz7ZKqMB5pdBivx3gtBnaYPky_x_jaxVTyjY-1GWNtSdqwc1TXC5JZmGSoyYWV0TyPDrIws2vB2YZEB67rtvbkmmKxjmC1s1OWMx4dTKfTEQ8xX9GOJ0v81Hkn1faqZQmDLAmDd7JgMgwomKfyoP9HNSBsldcmazih1WcrvdlaYMqE2oS21gsZXi_dGqMZ28hje7W_ExVq5cyWGlCGJw3N5dtMx6D5gA0KwgtEsbw67xTBEog1KcgSRNLEiG-WYFneSxxqBPtxU6YgErwhNuCsWa1BlKG8lm9NhZEgC4pk5xYdpELlb8aKHiTfQXQHdiDy54ehG0w9P5x5wXQ6D2xwAyIYwkP5n3ueA_3Qc517G9xqAucQQpnRwHHCuSuXhI4NCKbya_7VXPv69r__B73EbBI)

## Implementation

### Files Structure
- `model.py`: VQ-VAE architecture (Encoder, Decoder, Vector Quantizer)
- `utils.py`: Training utilities and visualization
- `VQ_VAE.py`: Training script and hyperparameter search
- `analysis.ipynb`: Results analysis and latent space visualization

### Hyperparameters
- Number of embeddings: [25, 50, 100]
- Embedding dimensions: [16, 32, 64]
- Training epochs: 50
- Beta (commitment loss): 0.25
- Optimizer: Adam

## Key Findings

### Latent Space Analysis
- Distinct codebook usage patterns across clothing categories
- Clear clustering of similar clothing items in t-SNE visualization
- Specialized codebook vectors for specific clothing features

### Reconstruction Quality
- Best performance on structured items (trousers, boots)
- Higher variability in complex items (shirts, pullovers)
- Successful capture of main clothing features and shapes

### Interpolation Results
- Smooth transitions between clothing categories
- Coherent intermediate representations
- Meaningful latent space organization

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

2. Analyze results:
```python
from utils import load_losses, plot_losses, find_best_model_params

# Load and visualize training results
losses = load_losses('./checkpoints/losses_dicts')
plot_losses(losses)
best_params = find_best_model_params(losses)
```

3. Run analysis notebook for detailed visualizations and insights
