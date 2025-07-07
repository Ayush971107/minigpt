# Transformer: Attention is All You Need

This repository contains a PyTorch implementation of the Transformer architecture as described in the seminal paper "Attention is All You Need" by Vaswani et al. (2017). The implementation focuses on a language modeling task, specifically trained on poetry data.

## 📝 Overview

The implementation includes all key components of the original Transformer architecture:

- Multi-Head Self-Attention mechanism
- Position-wise Feed-Forward Networks
- Residual connections and Layer Normalization
- Positional Encoding
- Masked self-attention for language modeling

## 🛠️ Implementation Details

### Model Architecture

1. **Token and Position Embeddings**
   - Learned token embeddings
   - Learned positional embeddings

2. **Multi-Head Attention**
   - Multiple attention heads in parallel
   - Scaled dot-product attention
   - Dropout for regularization

3. **Feed-Forward Network**
   - Two linear transformations with ReLU activation
   - Hidden dimension expansion (4x)

4. **Encoder-Decoder Structure**
   - Stacked self-attention layers
   - Residual connections around each sub-layer
   - Layer normalization

### Training

- **Dataset**: Gutenberg Poetry Dataset
- **Optimizer**: Adam
- **Learning Rate**: Configurable
- **Batch Size**: Configurable
- **Context Length**: Configurable block size

## 🚀 Getting Started

### Prerequisites

- Python 3.6+
- PyTorch
- tqdm (for progress bars)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Attention

# Install dependencies
pip install torch tqdm
```

### Training

1. Download and preprocess the poetry dataset:
   ```bash
   # This is handled automatically in the code
   # Downloads from http://static.decontextualize.com/gutenberg-poetry-v001.ndjson.gz
   ```

2. Run the training script:
   ```python
   python minigpt.py
   ```

### Generation

After training, you can generate text using the trained model:

```python
# Load the trained model
model = LanguageModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
```

## 📊 Model Configuration

The model can be configured using the following hyperparameters (defined at the top of `minigpt.py`):

- `batch_size`: Number of sequences processed in parallel
- `block_size`: Maximum context length
- `max_iters`: Number of training iterations
- `eval_interval`: How often to evaluate on validation set
- `learning_rate`: Learning rate for Adam optimizer
- `device`: Device to run on ('cuda' or 'cpu')
- `eval_iters`: Number of iterations to average for evaluation
- `n_embd`: Embedding dimension
- `n_head`: Number of attention heads
- `n_layer`: Number of transformer blocks
- `dropout`: Dropout rate

## 📈 Results

After training, the model learns to generate poetic text with reasonable coherence and structure. The quality of generations improves with longer training and larger model sizes.

## 📚 References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in neural information processing systems*, *30*.

2. The code is inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) implementation.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.