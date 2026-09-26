# Diabetes Prediction Using ML

A simple, clean, fully working academic machine learning web application prototype built for research article review and viva demonstration.

---

## 📌 Project Overview

This prototype demonstrates how a supervised Machine Learning algorithm (**Decision Tree Classifier**) can be trained on clinical benchmark data (**Pima Indians Diabetes Dataset**) and deployed as an interactive web tool using **Python Flask** and pure **HTML/CSS**.

- **Topic:** Diabetes Prediction Using ML
- **Context:** Machine Learning Research Article Review and Implementation
- **Architecture:** Monolithic Flask Web App with offline trained model (`diabetes_model.pkl`)

> **Academic Disclaimer:** This application is strictly an academic ML prototype for educational and research demonstration. It is **not** a clinically validated medical diagnosis tool and should never be used as a substitute for professional medical consultation.

---

## 🛠️ Technology Stack

- **Frontend:** Plain HTML5 & Vanilla CSS (No external CSS/JS frameworks like Bootstrap or Tailwind)
- **Backend:** Python (Flask framework)
- **Machine Learning:** Scikit-learn (`DecisionTreeClassifier`)
- **Data Handling:** Pandas, NumPy
- **Model Serialization:** Joblib

---

## 📁 Project Structure

```text
diabetes_prediction/
│
├── app.py                     # Flask web server and form handling
├── train_model.py             # Model training and evaluation script
├── diabetes.csv               # Pima Indians Diabetes dataset
├── diabetes_model.pkl         # Serialized trained Decision Tree model
├── requirements.txt           # Python package dependencies
├── README.md                  # Project documentation and viva guide
│
├── templates/
│   ├── index.html             # Homepage with input form and project info
│   └── result.html            # Result display page with patient summary
│
└── static/
    └── style.css              # Healthcare-themed vanilla CSS stylesheet
```

---

## 📊 Dataset & Features

The model is trained on `diabetes.csv` containing 768 patient records with 8 predictor features and 1 binary target:

| Feature Name | Description | Typical Unit / Range |
| :--- | :--- | :--- |
| **Pregnancies** | Number of times pregnant | Whole integer (0+) |
| **Glucose** | Plasma glucose concentration (2 hours in OGTT) | mg/dL (40 – 400) |
| **BloodPressure** | Diastolic blood pressure | mm Hg (0 – 140) |
| **SkinThickness** | Triceps skin fold thickness | mm (0 – 99) |
| **Insulin** | 2-Hour serum insulin | mu U/ml (0 – 846) |
| **BMI** | Body mass index | weight in kg / (height in m)² |
| **DiabetesPedigreeFunction** | Diabetes pedigree score (genetic predisposition) | Numerical score (0.05 – 2.5) |
| **Age** | Age in years | Years (1 – 120) |

**Target Column:** `Outcome`
- `0` = Non-Diabetic
- `1` = Diabetic

---

## 🚀 Step-by-Step Setup Instructions

### 1. Create a Virtual Environment (Optional but Recommended)
Open a terminal in the project directory:

```bash
python -m venv venv
```

Activate the virtual environment:
- **Windows (Command Prompt / PowerShell):**
  ```cmd
  venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
Run the model training script to train the Decision Tree and generate `diabetes_model.pkl`:

```bash
python train_model.py
```

Expected terminal output:
```text
Loading dataset...
Dataset shape: (768, 9)
Training Decision Tree...

Model Evaluation
-------------------------
Accuracy: ...
Precision: ...
Recall: ...
F1 Score: ...

Confusion Matrix:
[[... ...]
 [... ...]]

Model saved successfully:
.../diabetes_model.pkl
```

### 4. Run the Web Application
```bash
python app.py
```

### 5. Access the Web Application
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🎓 Academic Viva & Oral Examination Guide

During a viva or project defense, examiners typically test foundational understanding of ML and web integration. Here are answers to the key questions:

### 1. What is diabetes prediction?
Diabetes prediction is a binary classification task where machine learning models analyze patient clinical and demographic indicators (such as blood glucose, BMI, and age) to predict whether an individual belongs to the diabetic or non-diabetic class.

### 2. What is supervised learning?
Supervised learning is an ML paradigm where algorithms learn from labeled data. The training set consists of input pairs $(X, y)$, where $X$ are the feature vectors and $y$ are known ground-truth targets. The algorithm learns a mapping function $f(X) \approx y$.

### 3. What is a Decision Tree?
A Decision Tree is a non-parametric supervised learning algorithm structured like an upside-down tree with root, internal decision nodes, and terminal leaves. At each internal node, the dataset is split based on a feature value threshold that maximizes information gain or minimizes Gini impurity.

### 4. Why is a Decision Tree used for this prototype?
- **High Interpretability:** The decision rules can be easily visualized and understood as simple IF-THEN conditions.
- **Minimal Preprocessing:** It handles non-linear relationships without requiring feature scaling or normalization.
- **Pedagogical Value:** It is straightforward for students to explain in an academic presentation.
*(Note: Decision Trees are not universally the best algorithm for all datasets, but they serve as an interpretable baseline model).*

### 5. What are the input features?
8 health parameters: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, and Age.

### 6. What is the target variable?
`Outcome`, a binary classification target where `0` denotes non-diabetic and `1` denotes diabetic.

### 7. What is train/test split?
Train/test split divides available data into two distinct subsets:
- **Training set (80%):** Used by the algorithm to learn patterns and build the tree.
- **Testing set (20%):** Kept completely unseen during training to evaluate the model's true generalization performance.
- **Stratification (`stratify=y`):** Ensures both training and test sets contain the exact same proportion of diabetic and non-diabetic cases as the full dataset.

### 8. What is overfitting?
Overfitting occurs when a model learns the training data "too well," memorizing random noise and outliers rather than general patterns. An overfitted Decision Tree has near 100% training accuracy but performs poorly on new, unseen test data. Techniques like pruning or setting `max_depth` help control overfitting.

### 9. What is accuracy?
Accuracy is the proportion of total correct predictions (both diabetic and non-diabetic) over all predictions:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

### 10. What are precision, recall, and F1-score?
- **Precision:** Of all patients predicted as diabetic, how many actually are diabetic?
  $$\text{Precision} = \frac{TP}{TP + FP}$$
- **Recall (Sensitivity):** Of all patients who are truly diabetic, how many did the model correctly identify?
  $$\text{Recall} = \frac{TP}{TP + FN}$$
- **F1-Score:** The harmonic mean of precision and recall, providing a balanced metric especially when classes are imbalanced:
  $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 11. What is a confusion matrix?
A $2 \times 2$ table showing the breakdown of model predictions against ground truth:
- **True Positives ($TP$):** Diabetic classified as Diabetic.
- **True Negatives ($TN$):** Non-Diabetic classified as Non-Diabetic.
- **False Positives ($FP$):** Non-Diabetic incorrectly classified as Diabetic (Type I error).
- **False Negatives ($FN$):** Diabetic incorrectly classified as Non-Diabetic (Type II error, critical in healthcare).

### 12. How does the Flask backend communicate with the HTML form?
1. The user fills the HTML form inside `index.html`.
2. Clicking submit sends an HTTP `POST` request to the `/predict` route in `app.py`.
3. Flask retrieves submitted values using `request.form.get('FieldName')`.
4. Flask validates and processes the values, invokes the model, and passes results to `result.html` using Jinja2 templating (`render_template`).

### 13. How is the trained ML model loaded?
The model is serialized to disk once during `train_model.py` using `joblib.dump(model, 'diabetes_model.pkl')`. When Flask initializes `app.py`, it loads the file once using `joblib.load('diabetes_model.pkl')`, caching the model in memory. It does **not** retrain the model for every user request.

### 14. How is a prediction generated?
1. The 8 user-entered inputs are validated and converted into float numbers in the exact order expected by the model.
2. They are structured into a 2D array: `[[preg, gluc, bp, skin, ins, bmi, dpf, age]]`.
3. `model.predict([features])` evaluates the input against the decision nodes of the tree and returns `0` or `1`.
4. Flask renders `result.html` with the corresponding classification outcome.
