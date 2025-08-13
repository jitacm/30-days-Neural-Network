# Well-Rested Classification Model

Predicts whether individuals are well-rested based on health metrics using PyTorch. The enhanced MLP model significantly improves upon the original Perceptron's performance.

## Key Features
- 🧠 **Multi-Layer Perceptron** with 3 hidden layers
- ⚖️ **Automatic class balancing** for imbalanced datasets
- 📉 **Early stopping** with model checkpointing
- 📊 **Comprehensive visualizations**:
  - Training/validation loss curves
  - Confusion matrix
  - ROC curve
  - Probability distributions

## Requirements
- Python 3.7+
- Dependencies:
  ```bash
  pip install torch pandas scikit-learn matplotlib seaborn