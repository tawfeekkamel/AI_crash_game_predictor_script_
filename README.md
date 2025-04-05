# AI_crash_game_predictor_script_
Regression Model Comparison for Crash Multiplier Prediction  Compares RandomForestRegressor, MLPRegressor (Neural Network), and Ridge Regression using scikit-learn to predict a 'Multiplier(Crash)' value. Includes data preprocessing (scaling), training, R² score evaluation, and 5-fold cross-validation.
# Regression Model Comparison for Crash Multiplier Prediction

This repository contains a Python script that trains, evaluates, and compares three different regression models from the `scikit-learn` library to predict a 'Multiplier(Crash)' value.

## Description

The primary goal of this project is to demonstrate a machine learning workflow for a regression task. It involves:
1.  Loading data from a CSV file.
2.  Preprocessing the data (feature scaling).
3.  Splitting the data into training and testing sets.
4.  Training multiple regression models (`RandomForestRegressor`, `MLPRegressor`, `Ridge`).
5.  Evaluating the models using both a train/test split (R² score) and cross-validation.

## Features

* **Data Handling:** Loads data using `pandas`.
* **Preprocessing:** Applies `StandardScaler` for feature scaling.
* **Model Training:** Implements training for Random Forest, Multi-layer Perceptron (Neural Network), and Ridge Regression models.
* **Model Evaluation:**
    * Calculates the R² (Coefficient of Determination) score on a held-out test set.
    * Performs 5-fold cross-validation on the entire dataset for a more robust performance estimate.
* **Structured Code:** Uses Python classes (`Process`, `Train`, `Acc`) to organize the workflow stages.

## Models Used

* **Random Forest Regressor:** `sklearn.ensemble.RandomForestRegressor`
* **Multi-layer Perceptron Regressor:** `sklearn.neural_network.MLPRegressor`
* **Ridge Regression:** `sklearn.linear_model.Ridge`

## Dataset

* The script expects a CSV file named `data - Copy.csv` in the same directory.
* The target variable to predict is assumed to be in the column named `'Multiplier(Crash)'`.
* The `'Time'` column is dropped and not used as a feature.
* **(Optional: Add a more detailed description of your features and the 'Multiplier(Crash)' target here if available.)**

## Requirements

* Python 3.x
* pandas
* scikit-learn

## Installation
  Install the required libraries:
    ```bash
    pip install pandas scikit-learn
    ```

## Usage

1.  Make sure the `data - Copy.csv` file is present in the same directory as the script.
2.  Run the Python script from your terminal:
    ```bash
    python your_script_name.py
    ```
    *(Replace `your_script_name.py` with the actual name of your Python file)*

## Workflow

1.  **Data Processing (`Process` class):**
    * Loads the dataset.
    * Separates features (X) and the target variable (y).
    * Removes specified columns ('Multiplier(Crash)', 'Time') from features.
    * Applies `StandardScaler` to the features.
    * Splits the data into a training set (10%) and a test set (90%).
    * Initializes the three regression models (`RandomForestRegressor`, `MLPRegressor`, `Ridge`).
2.  **Cross-Validation (main script body):**
    * Performs 5-fold cross-validation (`cross_val_score`) for each initialized model on the *entire* scaled dataset (`process.X`, `process.y`) before the train/test split evaluation occurs.
    * Prints the individual fold scores and the mean cross-validation score for each model.
3.  **Training (`Train` class - triggered by `Acc`):**
    * Takes the processed data (specifically the 10% training split) and initialized models.
    * Fits (trains) each model on the training data (`train_X`, `train_y`).
4.  **Accuracy/Evaluation (`Acc` class):**
    * Re-initializes `Process` and `Train` (Note: This means data processing and model training happens again here based on the code structure).
    * Uses the trained models to make predictions on the test set (`test_X`).
    * Calculates the R² score comparing predictions to the actual test target values (`test_y`).
    * Prints the R² score (formatted as a percentage) for each model based on the 10%/90% split.

## Results

The script will output the following to the console:

1.  **Cross-Validation Scores:** For each model (Random Forest, MLP, Ridge), it will print the scores for each of the 5 folds and the mean cross-validation score. This gives an indication of how well the model is expected to perform on unseen data on average.
2.  **R² Scores:** For each model, it will print the R² score calculated on the 90% test set, formatted as a percentage (e.g., "The accuracy for RandomForestRegressor is XX.XX%."). This score reflects the model's performance specifically on that single train/test split.

## Notes
* any use to this program IAM NOT RESPONSIBLE FOR
* this is still beta and uder development 
* The current **train/test split** ratio is set to 10% for training and 90% for testing (`test_size=0.9`). This is an unusually small training set and might impact model performance. Consider adjusting this based on your dataset size and goals.
* The **MLPRegressor** is configured with `hidden_layer_sizes=(100, 50)` and `max_iter=3000`.
* **Ridge Regression** uses the default `alpha=1.0`.
* The `Acc` class currently re-runs the data processing and model training steps by creating new instances of `Process` and `Train`. This could be refactored for efficiency if needed.
