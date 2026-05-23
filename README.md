# Loan Default Prediction

## Project Overview
This project aims to predict loan defaults using a dataset containing various customer and loan-related features. The analysis involves data loading, extensive preprocessing, exploratory data analysis (EDA), and the application of several machine learning models to classify loan applications as either 'DEFAULT' or 'NO DEFAULT'.

## Dataset
The analysis uses `LoanDataset.csv`, which includes information such as customer age, income, home ownership, loan intent, loan amount, interest rate, and historical default status.

## Key Steps

### 1. Data Loading and Initial Inspection
- The dataset is loaded into a pandas DataFrame.
- Initial checks are performed to understand the data's shape and column data types.

### 2. Data Preprocessing
- **Missing Values**: Missing values in `customer_id`, `employment_duration`, `loan_int_rate`, `historical_default`, and `Current_loan_status` were identified. `historical_default` had a significant percentage of missing values (63.64%). Numerical missing values were imputed with the mean.
- **Data Type Conversion**: 
    - `loan_amnt` was cleaned by removing currency symbols (\£) and commas, then converted to a float.
    - `loan_int_rate` was converted to float.
    - `customer_income` was converted to float.
- **Target Variable Encoding**: The `Current_loan_status` column was converted into a binary format (1 for 'DEFAULT', 0 for 'NO DEFAULT').
- **Categorical Variable Encoding**: Categorical features such as `home_ownership`, `loan_intent`, `loan_grade`, and `historical_default` were converted to numerical representations using Label Encoding. Later, one-hot encoding was applied to these variables, and the `customer_id` was dropped.

### 3. Exploratory Data Analysis (EDA)
- **Distribution Analysis**: Visualizations (pie charts and bar plots) were created to understand the distribution of key features like `home_ownership`, `customer_age`, `loan_intent`, and `loan_grade`.
- **Target Variable Distribution**: A pie chart illustrated the breakdown of loan defaulters vs. non-defaulters.
- **Multivariate Analysis**: Heatmaps and stacked bar plots were used to explore the relationships between:
    - `home_ownership` and `Current_loan_status`
    - `loan_grade` and `Current_loan_status`
    - `loan_intent` and `Current_loan_status`

### 4. Model Training
The data was split into training and testing sets. Several classification models were trained and evaluated:
- **Random Forest Classifier**: Initial training and evaluation, followed by hyperparameter tuning using `GridSearchCV` and `StratifiedKFold` to find the best parameters and improve performance.
- **Decision Tree Classifier**: Trained and evaluated for comparison.
- **Naive Bayes Classifiers (GaussianNB and MultinomialNB)**: Trained and evaluated for comparison.
- **Multi-Layer Perceptron (MLPClassifier)**: A neural network model trained and evaluated.

## Technologies Used
- Python 3.12
- pandas (for data manipulation)
- numpy (for numerical operations)
- matplotlib (for plotting)
- seaborn (for statistical data visualization)
- scikit-learn (for preprocessing, model selection, and machine learning models)
- imbalanced-learn (for handling imbalanced datasets, specifically SMOTE)

## Usage
To run this analysis:
1. Ensure you have all the listed Python libraries installed (`pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn`).
2. Place the `LoanDataset.csv` file in the same directory as the notebook.
3. Execute the cells sequentially in the notebook.
