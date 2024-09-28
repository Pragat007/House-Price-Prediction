# House Price Prediction

This project involves predicting house prices using a dataset that contains various features related to housing characteristics. The dataset includes information like the area, the number of rooms, the year the house was sold, and many other features.

## Project Structure

- **Data**: The project utilizes a training dataset (`train.csv`) and a test dataset (`test.csv`) to build and evaluate the prediction model.
- **Libraries**: The following libraries are used:
  - `NumPy` for numerical operations
  - `Pandas` for data manipulation
  - `Matplotlib` and `Seaborn` for visualization
  - `Scikit-learn` for machine learning algorithms

## Dataset

- **Training data**: Contains features and the target variable (house prices). This file should be named `train.csv`.
- **Test data**: Contains the same features without the target variable. This file should be named `test.csv`.

Make sure that both `train.csv` and `test.csv` are placed in the root directory of the project before running the notebook.

## Steps

1. **Data Loading**: The training and test datasets are loaded using Pandas.
2. **Exploratory Data Analysis (EDA)**: Visualizations and summary statistics are used to explore the data.
3. **Data Preprocessing**: Handling missing values, feature engineering, and scaling of numerical features.
4. **Model Building**: 
   - Various machine learning algorithms are applied, such as:
     - Linear Regression
     - Random Forest
     - XGBoost
     - Lasso Regression
   - Hyperparameter tuning is performed using `RandomizedSearchCV` for Random Forest and XGBoost.
5. **Prediction**: The final model is used to predict house prices on the test dataset.

## Why These Models Were Used:

1. **Linear Regression**:
   - **Why**: It provides a simple, interpretable model to establish a baseline.
   - **Strength**: Fast to implement and interpret.
   - **Limitations**: May not perform well with non-linear relationships or complex feature interactions.

2. **Random Forest**:
   - **Why**: It models complex feature interactions and reduces overfitting by averaging multiple decision trees.
   - **Strength**: Handles high-dimensional and missing data, provides feature importance, and performs well on tabular data.
   - **Limitations**: Computationally expensive, especially for large datasets and extensive hyperparameter tuning.

3. **XGBoost**:
   - **Why**: Known for high performance, especially on structured/tabular data.
   - **Strength**: Fast, scalable, and effective with missing data and outliers.
   - **Limitations**: More complex to implement and prone to overfitting without careful tuning.

4. **Lasso Regression**:
   - **Why**: Regularizes the model and helps prevent overfitting by shrinking the coefficients of less important features.
   - **Strength**: Performs automatic feature selection by eliminating irrelevant features.
   - **Limitations**: May struggle with highly correlated features.

## Why Random Forest is Better:

- **Feature Interactions**: Can naturally model complex interactions between features.
- **Robustness**: Less sensitive to outliers and overfitting compared to linear models.
- **Feature Importance**: Offers insights into which features are most important for prediction.
- **Performance**: Provides better accuracy on non-linear relationships and is generally more robust than other models like Linear Regression.
- **Versatility**: Works well with both numerical and categorical data without requiring extensive preprocessing.

## Installation

To run the project locally, you need to install the following dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/Pragat007/house-price-prediction.git
   ```

2. Navigate to the project directory:
  ```bash
  cd house-price-prediction
  ```

3. Make sure you have the following files in the project directory:
  -  train.csv
  -  test.csv

4. Run the Jupyter Notebook:
  ```bash
  jupyter notebook house_price_prediction.ipynb
  ```

