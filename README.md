# Insurance Cost Analysis

## Project Overview
This project analyzes a medical insurance dataset to understand the factors affecting insurance charges and builds predictive models using various regression techniques. The analysis explores relationships between demographic factors (like age, gender, BMI) and lifestyle choices (like smoking) on insurance costs.

## Dataset
The dataset contains medical insurance information with the following features:
- Age
- Gender
- BMI (Body Mass Index)
- Number of children
- Smoker status
- Region
- Charges (target variable)

## Data Preprocessing
The following preprocessing steps were performed:
- Renamed columns for better readability
- Handled missing values:
  - Missing smoker status values were filled with the most frequent value
  - Missing age values were filled with the mean age
- Rounded charges to two decimal places for better presentation

## Exploratory Data Analysis
- Examined the relationship between smoking status and insurance charges
- Created visualizations using seaborn:
  - Regression plots to show correlation between variables
  - Box plots to display distribution of charges based on smoking status
- Calculated correlation matrix to identify relationships between features

## Machine Learning Models
Several regression models were implemented to predict insurance charges:

1. **Simple Linear Regression**
   - Using only smoker status as a predictor

2. **Multiple Linear Regression**
   - Using all available features as predictors

3. **Polynomial Regression with Preprocessing Pipeline**
   - Standardized data using StandardScaler
   - Applied polynomial feature transformation
   - Implemented linear regression on transformed features

4. **Ridge Regression**
   - Applied Ridge regression with alpha=0.1 to prevent overfitting
   - Used train-test split (80-20) to evaluate model performance

5. **Polynomial Ridge Regression**
   - Combined polynomial features (degree=2) with Ridge regression
   - Evaluated performance on test data using R² score

## Results
The polynomial Ridge regression model achieved the best performance, demonstrating the non-linear relationship between the predictors and insurance charges.

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## How to Run
1. Clone this repository
2. Install the required dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Run the Jupyter notebook or Python script

## Future Work
- Feature engineering to improve model performance
- Exploring other regression techniques (Random Forest, Gradient Boosting)
- Hyperparameter tuning to optimize model performance
- Deeper analysis of feature importance to understand key cost drivers
