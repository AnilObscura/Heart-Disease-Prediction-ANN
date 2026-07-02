# ❤️ Heart Disease Prediction using Artificial Neural Network (ANN)

> **A Deep Learning model built with TensorFlow/Keras to predict the presence of heart disease in patients based on clinical parameters. Achieved ~88% accuracy and an ROC-AUC score of 0.89.**

---

## Table of Contents
1. [Project Overview & Problem Statement](#project-overview--problem-statement)
2. [About the Dataset](#about-the-dataset)
3. [Tech Stack & Libraries](#tech-stack--libraries)
4. [How an Artificial Neural Network (ANN) Works](#how-an-artificial-neural-network-ann-works)
5. [Model Architecture & Training](#model-architecture--training)
6. [Evaluation & Results](#evaluation--results)
7. [Key Visualizations & Insights](#key-visualizations--insights)
8. [How to Run this Notebook](#how-to-run-this-notebook)
9. [License](#license)

---

## Project Overview & Problem Statement

Deep Learning aims to mimic the human brain through artificial neurons and multiple hidden layers. The most basic and foundational architecture of Deep Learning is the **Artificial Neural Network (ANN)**.

**Goal:** The objective of this project is to build a robust binary classification model using ANN to predict whether a person has heart disease or not, based on 13 medical attributes (such as age, cholesterol levels, resting blood pressure, etc.).

This project demonstrates the complete lifecycle of an ML project:
1. Data Loading & Analysis.
2. Data Cleaning & Preprocessing (Standardization).
3. Splitting data into Training and Testing sets.
4. Building a Sequential Neural Network.
5. Training the model using Backpropagation.
6. Evaluating the model using Confusion Matrix, ROC-AUC, and Accuracy metrics.

---

## About the Dataset

The dataset contains **303 patient records** with **13 independent clinical features** and **1 binary target column** (`0` = No Heart Disease, `1` = Heart Disease).

**Key Features Used for Prediction:**
- `age`: Age of the patient.
- `sex`: Gender (0 = Female, 1 = Male).
- `cp`: Chest Pain Type (0, 1, 2, 3).
- `trestbps`: Resting Blood Pressure (in mm Hg).
- `chol`: Serum Cholesterol (in mg/dl).
- `fbs`: Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False).
- `restecg`: Resting Electrocardiographic Results.
- `thalach`: Maximum Heart Rate Achieved.
- `exang`: Exercise Induced Angina (1 = Yes, 0 = No).
- `oldpeak`: ST depression induced by exercise relative to rest.
- `slope`: The slope of the peak exercise ST segment.
- `ca`: Number of major vessels (0-3) colored by fluoroscopy.
- `thal`: Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect).

**Target Variable:**
- `target`: Diagnosis of heart disease (1 = Disease present, 0 = No disease).

---

## Tech Stack & Libraries

| Layer | Technology |
| :--- | :--- |
| **Language** | Python 3.13+ |
| **Deep Learning Framework** | TensorFlow / Keras |
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning Utilities** | Scikit-learn (StandardScaler, train_test_split, confusion_matrix, roc_curve) |
| **Data Visualization** | Matplotlib, Seaborn |
| **Environment** | Google Colab / Jupyter Notebook |

---

## How an Artificial Neural Network (ANN) Works

An Artificial Neural Network (ANN) is inspired by the biological neurons in the human brain. It consists of interconnected layers of nodes (neurons) that process data.

### The Workflow:
1. **Input Layer:** Receives the raw data (the 13 clinical features).
2. **Weights & Biases:** Each connection between neurons has a "weight" that determines how important that input is. The neuron adds a "bias" to adjust the output.
3. **Activation Functions:** The weighted sum is passed through a non-linear activation function (like `ReLU` or `Sigmoid`). This decides if the neuron should "fire" or pass the information to the next layer.
4. **Hidden Layers:** The middle layers perform complex feature extraction. They find hidden patterns (like "high cholesterol + low max heart rate = high risk").
5. **Backpropagation:** During training, the model compares its prediction to the actual answer. It calculates the error (Loss) and sends that error backward through the network, tweaking the `weights` and `biases` to reduce the error.
6. **Output Layer:** The final layer produces the probability of the patient having heart disease (0 to 1).

This entire process is repeated over 100 epochs until the model's accuracy reaches a peak.

---

## Model Architecture & Training

This project utilizes a **Sequential Artificial Neural Network (ANN)** built using Keras.

### The Architecture:
- **Input Layer:** 13 neurons (representing the 13 medical features).
- **Hidden Layer 1:** 8 neurons, `ReLU` activation function, Uniform kernel initializer.
- **Hidden Layer 2:** 14 neurons, `ReLU` activation function, Uniform kernel initializer.
- **Output Layer:** 1 neuron, `Sigmoid` activation function (Outputs a probability between 0 and 1).

### Compilation & Training Hyperparameters:
- **Optimizer:** `Adam` (Adaptive Moment Estimation) - for efficient gradient descent.
- **Loss Function:** `Binary Crossentropy` (The standard loss function for binary classification).
- **Metrics:** `Accuracy`.
- **Batch Size:** `8` (Update weights after processing 8 samples).
- **Epochs:** `100` (Full passes over the training data).

### Data Preprocessing:
To ensure optimal neural network performance, the dataset features were scaled using `StandardScaler`. This transforms the data to have a mean of 0 and a standard deviation of 1, ensuring no single feature (like age) dominates others due to scale.

---

## Evaluation & Results
The Artificial Neural Network (ANN) was evaluated on unseen test data using a Confusion Matrix and multiple classification metrics.

---

## 🧩 Confusion Matrix

| Metric | Value | Description |
| :--- | :---: | :--- |
| **True Negatives (TN)** | **37** | Correctly predicted **No Disease** |
| **False Positives (FP)** | **7** | Incorrectly predicted **Disease** |
| **False Negatives (FN)** | **4** | Missed actual Disease cases |
| **True Positives (TP)** | **43** | Correctly predicted **Disease** |

---

## 🎯 Model Accuracy

The model accuracy is calculated using the standard classification formula:

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Substituting the values:

```text
Accuracy = (43 + 37) / (43 + 37 + 7 + 4)
         = 80 / 91
         = 87.91%
```

### ✅ Final Accuracy

**87.91%**

The Artificial Neural Network achieved an **87.91% classification accuracy** on unseen test data.

---

## 📈 ROC-AUC Score

The ROC-AUC Curve evaluates the trade-off between the **True Positive Rate (Sensitivity)** and the **False Positive Rate (1 − Specificity).**

### ✅ ROC-AUC Score

**0.89**

An **AUC score of 0.89** indicates that the model has **excellent discriminative ability** in distinguishing between patients with and without heart disease.

---

# 📸 Model Visualizations & Insights

The following visualizations provide deeper insight into the model's performance and the dataset.

---

## 1️⃣ Model Training Performance (Loss & Accuracy)

Training and validation curves over 100 epochs demonstrate successful model convergence. Minor fluctuations in validation accuracy indicate slight overfitting during training, while the final test performance remains strong.

![Training Graphs](https://github.com/AnilObscura/Heart-Disease-Prediction-ANN/blob/0df6aee40cf363e9f17e33772771fba080f5f9a9/s1.PNG)

---

## 2️⃣ Feature Correlation Heatmap

The correlation heatmap highlights relationships between different clinical features. Variables such as **Chest Pain Type (`cp`)** and **Maximum Heart Rate (`thalach`)** show strong correlations with heart disease prediction.

![Correlation Heatmap](https://github.com/AnilObscura/Heart-Disease-Prediction-ANN/blob/0df6aee40cf363e9f17e33772771fba080f5f9a9/s2.PNG)

---

## 3️⃣ Exploratory Data Analysis (Boxplots)

Boxplots compare feature distributions between healthy patients and patients diagnosed with heart disease. The visualization indicates that patients with heart disease generally achieve a higher maximum heart rate.

![EDA Boxplots](https://github.com/AnilObscura/Heart-Disease-Prediction-ANN/blob/782466ea9e60ad8a980836a1016d2947737744ed/s3a.PNG)
![EDA Boxplots](https://github.com/AnilObscura/Heart-Disease-Prediction-ANN/blob/0df6aee40cf363e9f17e33772771fba080f5f9a9/s3b.PNG)

---

## 4️⃣ ROC-AUC Curve

The Receiver Operating Characteristic (ROC) Curve illustrates the model's classification capability across different threshold values.

With an **AUC score of 0.89**, the model demonstrates excellent classification performance.

![ROC Curve](https://github.com/AnilObscura/Heart-Disease-Prediction-ANN/blob/0df6aee40cf363e9f17e33772771fba080f5f9a9/s4.PNG)

---

# 🚀 How to Run Locally

Follow these steps to run the project on your local machine.

---

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/AnilObscura/Heart-Disease-Prediction-ANN.git
cd Heart-Disease-Prediction-ANN
```

---

## 2️⃣ Create and Activate a Python Virtual Environment (Recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn jupyter
```

---

## 4️⃣ Open the Jupyter Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook Heart_Disease_Prediction_using_ANN.ipynb
```

Alternatively, you can open the notebook directly in **Visual Studio Code** or upload it to **Google Colab**, where TensorFlow is pre-installed.

---

## 5️⃣ Dataset

Ensure the dataset file

```text
Dataset--Heart-Disease-Prediction-using-ANN.csv
```

is placed in the same directory as the notebook.

If the dataset is stored elsewhere, update the dataset path in the first notebook cell accordingly.

---

# 📄 License

This project is licensed under the **MIT License**.
