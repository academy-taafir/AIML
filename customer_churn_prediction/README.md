# 1. Customer Churn Prediction
- Univariate: Distribution of tenure, monthly charges, contract type.
- Bivariate: contract type vs. churn, monthly charges vs. churn.
- ML Use: Train a classification model to predict if a customer will churn.

## Univariate Analysis (Individual Variable Exploration)
### 1. Distribution of tenure
  - Plot a histogram or KDE plot.
  - Look for common tenure values — are many users new (<12 months) or long-term customers?

### 2. Distribution of monthly_charges
  - Check min, max, mean, and standard deviation.
  - Use histograms or box plots to detect outliers and skewness.

### 3. Frequency of contract_type
  - Use a bar chart to show counts of each contract type.
  - Helps see which plan is most common.

### 4. Churn Rate Overview (churn)
  - Calculate overall churn rate (percentage of True values).
  - Important baseline for classification.

## Bivariate Analysis (Pairwise Relationships)
### 5. Churn vs. Contract Type
  - Create a grouped bar plot or stacked bar chart.
  - Determine if month-to-month contracts have a higher churn rate.

### 6. Monthly Charges vs. Churn
  - Use box plots or violin plots to compare charge distributions for churned vs. retained users.
  - Higher charges may correlate with higher churn.

### 7. Tenure vs. Churn
  - Plot average tenure grouped by churn status.
  - Short tenure customers may churn more.

## Statistical & Structural Checks
### 8. Check for Missing Values
  - df.isnull().sum()
  - Even though synthetic data is clean, always validate for real-world cases.

### 9. Check for Imbalanced Target Variable
  - Is churn heavily skewed towards False or True?
  - May need resampling techniques for ML (like SMOTE or class weights).

### 10. Correlation Matrix / Cramér’s V
  - Use correlation for numeric variables (tenure, monthly_charges).
  -For categorical vs categorical (e.g., contract_type vs. churn), use Cramér’s V or chi-square test.
