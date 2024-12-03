# deep-learning-challenge
# Alphabet Soup Charity Deep Learning Model

## Overview
This project involves building and optimizing a deep learning model to predict the success of donations made to the Alphabet Soup Charity. The goal was to preprocess the dataset, build a robust neural network model, and optimize it to achieve an accuracy level above 75%.

## Dataset Description
- **Source**: Alphabet Soup Charity dataset
- **Target Variable**: `IS_SUCCESSFUL` - Indicates whether a donation was successful (1) or not (0).
- **Features**: Various attributes related to application types, classifications, and other donation details.
- **Removed Variables**: `EIN` and `NAME` were excluded as they do not contribute to prediction.

## Project Steps
1. **Data Preprocessing**:
    - Removed irrelevant columns (`EIN`, `NAME`).
    - Encoded categorical variables using `pd.get_dummies()`.
    - Scaled numeric features using `StandardScaler`.

2. **Model Design**:
    - A sequential deep learning model with the following architecture:
        - **Input Layer**: Matches the number of features after preprocessing.
        - **Hidden Layers**:
            - Layer 1: 80 neurons, ReLU activation.
            - Layer 2: 40 neurons, ReLU activation.
            - Layer 3 (in Model 2 and 3): 20 neurons, ReLU activation.
        - **Output Layer**: 1 neuron, Sigmoid activation for binary classification.

3. **Training and Evaluation**:
    - Used `binary_crossentropy` as the loss function and `Adam` optimizer.
    - Implemented `EarlyStopping` to prevent overfitting.
    - Achieved a final accuracy of **~75.1%** in Model 3.

4. **Optimization Techniques**:
    - Adjusted neurons and added dropout regularization.
    - Tuned learning rate, epochs, and batch size.

## Results
- Final model achieved an accuracy of ~75.1%.
- Successfully met the target performance threshold.

## Recommendations
1. **Alternative Models**:
    - Experiment with Random Forest, Gradient Boosting, or XGBoost for structured data.
2. **Feature Engineering**:
    - Explore additional feature creation or dimensionality reduction techniques.
3. **Hyperparameter Tuning**:
    - Use tools like Keras Tuner or Optuna for more systematic parameter optimization.
4. **Ensemble Approaches**:
    - Combine multiple models to leverage their strengths.

## How to Run the Project
1. Install required dependencies:
    ```
    pip install tensorflow pandas scikit-learn
    ```
2. Load the dataset and preprocess it as described in the notebook.
3. Run the `AlphabetSoupCharity` notebook to train and evaluate the model.

## Files in this Repository
- `AlphabetSoupCharitymodel3optimizer.ipynb`: Notebook with the optimized model (Model 3).
- `AlphabetSoupCharity-Copy1.ipynb`: Notebook with initial experiments and Model 1.
- `AlphabetSoupCharity model 2optimizer.ipynb`: Notebook with Model 2 experiments.
- `README.md`: This file, providing an overview of the project.
