# 🩺 Breast Cancer Classification with K-NN

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)

Machine learning model using K-Nearest Neighbors (K-NN) algorithm for binary classification of breast cancer cells as malignant or benign using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

## 🎯 Objective

Develop a robust classification system to assist in early breast cancer detection by analyzing cell nucleus characteristics from digitized images of fine needle aspirate (FNA) samples.

## 📈 Dataset

**Wisconsin Diagnostic Breast Cancer (WDBC)**
- **Samples**: 569 instances
- **Features**: 30 real-valued features computed from cell nucleus images
- **Classes**: 
  - **M** (Malignant): 212 cases
  - **B** (Benign): 357 cases
- **Source**: UCI Machine Learning Repository

### Feature Categories

For each cell nucleus, 10 characteristics are measured:
1. Radius
2. Texture
3. Perimeter
4. Area
5. Smoothness
6. Compactness
7. Concavity
8. Concave points
9. Symmetry
10. Fractal dimension

For each characteristic, three values are computed:
- Mean
- Standard error
- "Worst" (largest value)

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/masoud-rafiee/breast-cancer-analysis-knn.git
cd breast-cancer-analysis-knn
pip install -r requirements.txt
```

### Usage

**Run Jupyter Notebook:**
```bash
jupyter notebook "k-NN Breast Cancer.ipynb"
```

**Run Python script:**
```bash
python "k-NN Breast Cancer.py"
```

## 🛠️ Model Architecture

### K-Nearest Neighbors (K-NN)

- **Algorithm**: Distance-based lazy learning
- **Distance Metric**: Euclidean distance
- **Optimal K**: Determined via cross-validation (typically k=5-11)
- **Data Preprocessing**: 
  - Feature scaling (StandardScaler)
  - Train-test split (80/20)

### Pipeline

```python
1. Load WDBC dataset (wdbc.data)
2. Exploratory Data Analysis (EDA)
3. Feature scaling & normalization
4. Train-test split
5. K-NN model training
6. Hyperparameter tuning (k selection)
7. Model evaluation (accuracy, precision, recall, F1)
8. Confusion matrix analysis
```

## 📄 Project Structure

```
breast-cancer-analysis-knn/
├── k-NN Breast Cancer.ipynb   # Interactive analysis
├── k-NN Breast Cancer.py      # Standalone script
├── wdbc.data                  # WDBC dataset (569 samples)
├── wdbc.names                 # Feature descriptions
├── HW3.pdf                    # Assignment report
├── requirements.txt           # Dependencies
└── README.md
```

## 📊 Model Performance

### Expected Results

| Metric       | Score   |
|--------------|--------:|
| **Accuracy** | ~96-97% |
| **Precision**| ~95%    |
| **Recall**   | ~97%    |
| **F1-Score** | ~96%    |

### Confusion Matrix Interpretation

- **True Positives (TP)**: Correctly identified malignant cases
- **True Negatives (TN)**: Correctly identified benign cases
- **False Positives (FP)**: Benign misclassified as malignant (Type I error)
- **False Negatives (FN)**: Malignant misclassified as benign (Type II error - critical to minimize)

## 📚 Key Insights

1. **Feature Importance**: "Worst" values (max measurements) are strong predictors
2. **Optimal K**: Cross-validation shows k=7-11 performs best
3. **Clinical Relevance**: High recall prioritized to minimize false negatives
4. **Scalability**: Fast inference time suitable for real-time diagnostics

## ⚠️ Limitations

- K-NN is sensitive to feature scaling
- Computational cost increases with dataset size
- No probabilistic interpretation of predictions
- Imbalanced dataset (more benign than malignant samples)

## 🔮 Future Enhancements

- [ ] Ensemble methods (Random Forest, Gradient Boosting)
- [ ] Deep learning approaches (MLP, CNN)
- [ ] Feature selection using PCA or SelectKBest
- [ ] Hyperparameter optimization with GridSearchCV
- [ ] SMOTE for handling class imbalance
- [ ] Model deployment with Flask API

## 📜 References

- [WDBC Dataset - UCI Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- Wolberg, W.H., Street, W.N., & Mangasarian, O.L. (1995). *Machine learning techniques to diagnose breast cancer from fine-needle aspirates*

## 👤 Author

**Masoud Rafiee**  
GitHub: [@masoud-rafiee](https://github.com/masoud-rafiee)  
LinkedIn: [masoud-rafiee](https://linkedin.com/in/masoud-rafiee)

## 📄 License

MIT License

---

*This project demonstrates supervised learning for medical diagnostics using classical ML algorithms.*
