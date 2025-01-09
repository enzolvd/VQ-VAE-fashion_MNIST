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
[![](https://mermaid.ink/img/pako:eNqdVetumzAUfhXk_iVdMZcQNPXHmk2q1Glrp-3HSFU52CRWwKbGpDf1OfZAe7H5AoSkbdrVQtbh8J3vHM7FfgAZxwQkIC_4TbZEQjpnFzPmqFU384VA1dI5ZVUjrU4vTAXJJOWsR-pFNSg1UOe0RAviwNi5tZt3aXGE4RnbIf_MdADiNfqMs7WXfqGils6Jkp1PBc9Wzse5-HA88BQFlxsbQWovvSD1G6CY3zAvnaq9RmVVkBbtBTp6s8F4mxn2zPuhmhkOmFs0NEC1kXJOMKZscYVpue3CT78LMrpuEJPOjrO95s-l-df56xnGZM75Kj1phTYHbH61VnAu6r9_XgrWxJie653eI029J5QpMRV3ngvo4myj5aqfvjVSN5SuuIlm2FPOaHSs0xTtVGKD8eF2QiNj0lQw_VkNqwHjW_VsoRWo4w93e2hT6t12C1sHXu9gv4lCdl6CJ176Gj_pvcAYUUb1wFFJUWFn4mXD5-pgc_uGKlSbQrxhss1BYAI0M2uVRux-tdVpyajM8FmdETsc7HGwx8ENri-Qb3U92jcfTE923tuO3tHb3Q5Yl1Gr4-0v8DZHLU-B6npKcgeTHDWFdHJaFMlBPsndWgq-IsmB7_utPLqhWC4TWN26GS-4SA6Ojo52eIg9_DqefB5l3jup1tctyyTO55P4nSyYDAOKx5n60f-jGhC2rdAma_jBtIOrG8A1FdciNCJ0TQHJ8FbobGwRXVuv9fW2Jyq1ZeCqkdJCqATDFbncxGD4gAtKIkpEsbrxHjTBDMglKckMJErESKxmYMYeFQ41kv-4YxlIpGiICwRvFkuQ5Kio1VtTYSTJlCI1SmWvrRD7zfnWO0gewC1IoslhNAniMBj7kQ9jD7rgDiRQqX0_hmEU--HEH4ePLrg3BN4hHEdR4E9gGKgVjUMXEEzVKfzVXtfm1n78B8epWHU?type=png)](https://mermaid.live/edit#pako:eNqdVetumzAUfhXk_iVdMZcQNPXHmk2q1Glrp-3HSFU52CRWwKbGpDf1OfZAe7H5AoSkbdrVQtbh8J3vHM7FfgAZxwQkIC_4TbZEQjpnFzPmqFU384VA1dI5ZVUjrU4vTAXJJOWsR-pFNSg1UOe0RAviwNi5tZt3aXGE4RnbIf_MdADiNfqMs7WXfqGils6Jkp1PBc9Wzse5-HA88BQFlxsbQWovvSD1G6CY3zAvnaq9RmVVkBbtBTp6s8F4mxn2zPuhmhkOmFs0NEC1kXJOMKZscYVpue3CT78LMrpuEJPOjrO95s-l-df56xnGZM75Kj1phTYHbH61VnAu6r9_XgrWxJie653eI029J5QpMRV3ngvo4myj5aqfvjVSN5SuuIlm2FPOaHSs0xTtVGKD8eF2QiNj0lQw_VkNqwHjW_VsoRWo4w93e2hT6t12C1sHXu9gv4lCdl6CJ176Gj_pvcAYUUb1wFFJUWFn4mXD5-pgc_uGKlSbQrxhss1BYAI0M2uVRux-tdVpyajM8FmdETsc7HGwx8ENri-Qb3U92jcfTE923tuO3tHb3Q5Yl1Gr4-0v8DZHLU-B6npKcgeTHDWFdHJaFMlBPsndWgq-IsmB7_utPLqhWC4TWN26GS-4SA6Ojo52eIg9_DqefB5l3jup1tctyyTO55P4nSyYDAOKx5n60f-jGhC2rdAma_jBtIOrG8A1FdciNCJ0TQHJ8FbobGwRXVuv9fW2Jyq1ZeCqkdJCqATDFbncxGD4gAtKIkpEsbrxHjTBDMglKckMJErESKxmYMYeFQ41kv-4YxlIpGiICwRvFkuQ5Kio1VtTYSTJlCI1SmWvrRD7zfnWO0gewC1IoslhNAniMBj7kQ9jD7rgDiRQqX0_hmEU--HEH4ePLrg3BN4hHEdR4E9gGKgVjUMXEEzVKfzVXtfm1n78B8epWHU)
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
