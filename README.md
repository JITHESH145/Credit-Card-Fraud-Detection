# Credit Card Fraud Detection Using Classification Algorithms

This project is a comprehensive implementation of a machine learning pipeline in Python to detect fraudulent credit card transactions. The primary challenge addressed is the highly imbalanced nature of the dataset, where fraudulent transactions are extremely rare compared to legitimate ones.

## Project Overview

The core objective is to build and evaluate multiple supervised classification models to accurately identify fraudulent activities. The workflow covers the entire machine learning pipeline, from data preprocessing to model evaluation and selection.

### Key Features & Methodology

*   **Data Preprocessing**: Standardized the `Time` and `Amount` features using `StandardScaler` to ensure all features contribute equally to the model's performance.
*   **Handling Class Imbalance**: Implemented the **Synthetic Minority Over-sampling Technique (SMOTE)** on the training data to create a balanced dataset, preventing model bias towards the majority class.
*   **Model Training**: Trained and compared six different classification algorithms:
    *   Logistic Regression
    *   K-Nearest Neighbors (KNN)
    *   Decision Tree
    *   Random Forest
    *   Support Vector Machine (SVM)
    *   XGBoost
*   **Model Evaluation**: Assessed each model's performance on the original, imbalanced test set using metrics crucial for fraud detection:
    *   **Confusion Matrix**
    *   **Precision**
    *   **Recall** (prioritized to minimize missed fraud cases)
    *   **F1-Score**
*   **Model Selection**: The final model was selected based on its superior ability to identify fraudulent transactions, with a primary focus on achieving high recall.

## Technologies Used

*   **Python**
*   **Pandas**: For data manipulation and analysis.
*   **NumPy**: For numerical operations.
*   **Scikit-learn**: For data preprocessing, model training, and evaluation.
*   **Imbalanced-learn**: For implementing the SMOTE technique.
*   **Matplotlib & Seaborn**: For data visualization and plotting confusion matrices.
*   **XGBoost**: For the XGBoost classifier.
