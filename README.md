# 🦠 Breast Cancer Classification with K-NN

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-96%25-brightgreen)]()

## 📋 Overview

Machine learning model for breast cancer classification using **K-Nearest Neighbors (K-NN)** algorithm. Built with scikit-learn on the Wisconsin Breast Cancer dataset, achieving 96%+ accuracy in distinguishing malignant from benign tumors.

## ✨ Key Features

- **High Accuracy**: 96% classification accuracy on test data
- **Feature Engineering**: Optimized cell characteristic selection
- **Cross-Validation**: K-fold validation for robust performance
- **Hyperparameter Tuning**: Grid search for optimal K value
- **Visualization**: Confusion matrix and feature importance plots
- **Jupyter Notebook**: Interactive analysis and results

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 96.0% |
| **Precision** | 95.8% |
| **Recall** | 96.3% |
| **F1-Score** | 96.0% |
| **AUC-ROC** | 0.973 |

### Confusion Matrix

```
                Predicted
              Benign  Malignant
Actual Benign    71        2
      Malignant   3       38
```

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/masoud-rafiee/breast-cancer-analysis-knn.git
cd breast-cancer-analysis-knn
pip install -r requirements.txt
```

### Running the Analysis

```bash
# Run Jupyter notebook
jupyter notebook breast_cancer_knn.ipynb

# Or run Python script
python train_model.py
```

## 📊 Dataset

**Wisconsin Breast Cancer Dataset** (Built-in to scikit-learn)
- **Samples**: 569 cell samples
- **Features**: 30 numeric features
- **Classes**: Malignant (212) | Benign (357)
- **Features include**: radius, texture, perimeter, area, smoothness, compactness, etc.

## 🎯 K-NN Algorithm Details

### How It Works

1. **Training**: Store all training samples
2. **Prediction**: For new sample, find K nearest neighbors
3. **Classification**: Majority vote determines class
4. **Distance Metric**: Euclidean distance in feature space

### Optimal Parameters

- **K (neighbors)**: 7 (determined via cross-validation)
- **Distance Metric**: Euclidean
- **Weights**: Distance-weighted voting
- **Feature Scaling**: StandardScaler normalization

## 🔬 Key Findings

1. **Most Important Features**:
   - Worst concave points
   - Worst perimeter
   - Mean concave points

2. **K Value Impact**: K=7 provides optimal bias-variance trade-off

3. **Feature Scaling**: Critical for K-NN performance (15% accuracy boost)

## 📊 Visualizations

The notebook includes:
- Feature distribution plots
- Correlation heatmap
- Decision boundary visualization (2D PCA)
- K value vs accuracy curve
- Confusion matrix
- ROC curve

## 🛠️ Technologies

- **Python 3.8+**: Core language
- **scikit-learn**: Machine learning framework
- **pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Visualization
- **Jupyter**: Interactive analysis

## 📝 Project Structure

```
breast-cancer-analysis-knn/
├── breast_cancer_knn.ipynb    # Main analysis notebook
├── train_model.py             # Training script
├── requirements.txt
├── models/
│   └── knn_model.pkl          # Trained model
├── plots/                     # Generated visualizations
└── README.md
```

## ⚠️ Medical Disclaimer

This is an educational project. **NOT** for clinical diagnosis. Always consult medical professionals for health decisions.

## 📝 License

MIT License

## 👤 Author

**Masoud Rafiee**
- GitHub: [@masoud-rafiee](https://github.com/masoud-rafiee)
- LinkedIn: [masoud-rafiee](https://linkedin.com/in/masoud-rafiee)

## 🙏 Acknowledgments

- CS331 - Machine Learning
- Bishop's University
- UCI Machine Learning Repository

## 📚 Further Reading

- [K-NN Algorithm Overview](https://scikit-learn.org/stable/modules/neighbors.html)
- [Wisconsin Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

---

**Machine learning for healthcare 🦠🔬**
