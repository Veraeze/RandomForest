# RandomForest
##  Random Forest Classifier —> Theory Summary

###  What is Random Forest?
Random Forest (also called Random Decision Forest) is an ensemble learning method used for classification and regression tasks. It operates by constructing multiple decision trees during the training phase and outputs the **majority vote** (in classification) or **average prediction** (in regression) from those trees.

In simple terms, Random Forest builds several decision trees and combines their results to improve accuracy and reduce overfitting.


###  Why Use Random Forest?

- **No Overfitting:** By combining multiple decision trees, the model generalizes better than a single tree.
- **High Accuracy:** Aggregating predictions often results in improved accuracy.
- **Efficient on Large Datasets:** Scales well to large datasets with high dimensionality.
- **Handles Missing Data:** Can maintain performance even when some features are missing.
- **Estimates Missing Values:** Can fill in missing values during training.


###  Real-World Applications of Random Forest

1. **Remote Sensing (ETM Devices):** Used in Earth observation systems to classify satellite images.
2. **Object Detection:** Useful in detecting multiple objects in complex environments (e.g., traffic).
3. **Kinect Gaming Console:** Random Forest algorithms are used to track body movements and mirror them in gameplay.


###  Relationship with Decision Trees

Random Forest is built on the foundation of Decision Trees. It enhances the predictive performance of decision trees by combining the output of many trees trained on random subsets of data.


###  Example Use Case: Iris Flower Classification

The theory was illustrated using the **Iris Flower Dataset**, which includes:
- **Target Classes:** Setosa, Versicolor, Virginica
- **Features:**
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width

The goal is to classify the species of an iris flower based on its physical attributes.





#  Iris Flower Classification using Random Forest

This project uses the Random Forest Classifier to predict the species of iris flowers based on petal and sepal dimensions.


##  Dataset Overview
- **Dataset**: Iris Dataset (from `sklearn.datasets`)
- **Features**:
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- **Target**: Species (Setosa, Versicolor, Virginica)


##  Steps Taken

### 1.  Import Libraries
Standard data science libraries were imported: `NumPy`, `Pandas`, `Seaborn`, `Matplotlib`, and `scikit-learn`.

### 2.  Data Loading & Inspection
- Loaded the iris dataset from `sklearn.datasets`.
- Created a DataFrame using the features.
- Inspected the structure, basic statistics, and shape of the data.

### 3. ️ Added Target Labels
- Mapped numeric target values to categorical species names (Setosa, Versicolor, Virginica).

### 4.  Data Splitting
- Used a custom method to split data into training and testing sets (75% training, 25% testing) using `np.random.uniform`.

### 5.  Target Preparation
- Used `pd.factorize()` to convert species names in training data to numerical format.

### 6.  Model Building
- Created and trained a **RandomForestClassifier** with:
  - `n_jobs=2`
  - `random_state=0`

### 7.  Model Prediction
- Predicted species for test samples.
- Displayed predicted class names and class probabilities.
- Compared predicted species with actual values.

### 8.  Evaluation
- **Confusion Matrix**:
  
  | Actual \ Predicted | Setosa | Versicolor | Virginica |
  |--------------------|--------|------------|-----------|
  | Setosa             |   13   |     0      |     0     |
  | Versicolor         |   0    |     5      |     2     |
  | Virginica          |   0    |     0      |    12     |

- **Accuracy**: `93.75%`
- **Classification Report**:
  - Setosa: Precision: 1.00, Recall: 1.00
  - Versicolor: Precision: 1.00, Recall: 0.71
  - Virginica: Precision: 0.86, Recall: 1.00
  - Macro avg F1-score: 0.92


##  Insights
- The model performed **perfectly** for Setosa and Virginica.
- Most errors occurred in predicting Versicolor.
- Random Forest proved to be a powerful classifier for this multi-class problem with strong generalization.


##  Conclusion
This project shows the power of Random Forest for multi-class classification tasks, especially in biologically meaningful datasets like the Iris flower dataset.







# Bank Customer Churn Prediction

This project uses the Random Forest Classifier to predict customer churn in a bank dataset. The aim is to identify customers likely to leave the bank (churn) based on features like age, balance, credit score, and other customer attributes.

##  Dataset Overview

- **Rows:** 10,000
- **Target Variable:** `churn` (0 = No churn, 1 = Churn)
- **Features Include:**
  - Numerical: credit_score, age, balance, estimated_salary, etc.
  - Categorical: country, gender
  - Binary: credit_card, active_member

##  Objective

To build a machine learning model that:
- Accurately predicts customer churn
- Improves recall on the churn class (class 1) due to class imbalance
- Provides insights into which features most influence churn

##  Technologies Used

- Python, Jupyter Notebook
- pandas, numpy, seaborn, matplotlib
- scikit-learn

##  Workflow Summary

### 1. Data Loading and Exploration
- Imported the dataset using `pandas`
- Displayed data shape, structure, and statistical summary
- Visualized class distribution using `countplot`

### 2. Handling Class Imbalance
- Observed class imbalance from visual inspection (churn: ~20%, no churn: ~80%)
- Strategy to improve recall included:
  - Tuning classification threshold
  - Monitoring confusion matrix metrics

### 3. Preprocessing
- Encoded categorical variables using `ColumnTransformer`:
  - OneHotEncoder for `country`
  - OrdinalEncoder for `gender`
- Split dataset using `train_test_split`

### 4. Exploratory Visualizations
- Correlation heatmap on encoded features
- Boxplots:
  - `age` vs `churn` (showed distinct distribution patterns)
  - `active_member` vs `churn`
- Pie charts:
  - Distribution of churn across `gender`
  - Distribution of churn by `country`

### 5. Model Building
- Built pipeline with:
  - Preprocessing (`ColumnTransformer`)
  - Random Forest Classifier (`RandomForestClassifier`)
- Trained model on training set and evaluated on test set

### 6. Evaluation
- **Initial Accuracy:** ~86.3%
- **Confusion Matrix:**
[[1523   72]
[ 202  203]]
- **Recall on churn class (1):** 0.50

### 7. Recall Improvement Experiment
- Tuned the prediction threshold from default 0.5 to a lower value to boost recall
- Best results observed:
- **Accuracy:** 82.6%
- **Recall (churn class):** 0.69
- **Precision (churn class):** 0.56

### 8. Postmodel Visualizations
- Feature importance bar plot (most important: `age`, `balance`, `credit_score`)
- Confusion matrix heatmap (before and after threshold tuning)


##  Conclusion

- **Random Forest** delivered strong overall accuracy and interpretability.
- **Threshold tuning** significantly improved recall, essential for identifying churned customers.
- Feature analysis revealed `age`, `active_member`, and `balance` as critical indicators.







#  Loan Default Prediction 

This project focuses on predicting whether a customer will default on a loan using the Random Forest Classifier. The dataset is highly imbalanced, and various techniques were applied to improve recall on the minority class (defaults).


##  Dataset Overview

- **Total rows**: 255,347  
- **Target column**: `Default` (0 = No Default, 1 = Default)  
- **Problem type**: Binary Classification  
- **Class Imbalance**:  
  - No Default (0): ~230,000  
  - Default (1): ~25,000


##  Data Preparation

- Loaded data and checked for duplicates  
- Visualized distribution of target and features  
- Split data into training and test sets  
- Categorical features separated into:
  - **OneHotEncoded**: `Employment`, `MaritalStatus`, `LoanPurpose`, `HasCoSigner`
  - **OrdinalEncoded**: `HasMortgage`, `HasDependents`, `Education`


##  Exploratory Data Analysis

- Countplots and boxplots for all major features by target class  
- Violin plot for credit score density  
- Pie charts for `LoanPurpose` and `EmploymentType`  
- Heatmap of numeric feature correlations  
- Insights confirmed class imbalance and feature influence on defaults


## ️ Modeling Steps

1. **Random Forest Classifier**  
   - Pipeline built using `ColumnTransformer` and `RandomForestClassifier`  
   - Initial model showed very high accuracy but poor recall on defaults

2. **SMOTE (Synthetic Minority Over-sampling Technique)**  
   - Applied to balance classes before training  
   - Recall improved slightly but remained low due to extreme imbalance

3. **Hyperparameter Tuning**  
   - Tried different combinations (n_estimators, max_depth, etc.)  
   - Recall for class 1 remained minimal

4. **Threshold Tuning**  
   - Predicted probabilities from model  
   - Evaluated multiple thresholds:
     - **Threshold = 0.25 chosen**  
       - Precision (class 1): 0.36  
       - Recall (class 1): 0.27  
       - Balanced trade-off


##  Final Results at Threshold 0.25

- **Accuracy**: 86%  
- **Confusion Matrix**:
[[63364  4309]
[ 6492  2440]]
- **Recall (Class 1)**: 0.27  
- **Precision (Class 1)**: 0.36  
- **F1-score (Class 1)**: 0.31  
- **ROC Curve**: Plotted to visualize probability threshold effects


## Conclusion

- Imbalanced datasets require more than just accuracy for evaluation  
- SMOTE and threshold tuning are powerful for improving minority class performance  
- Even small recall improvements can be meaningful in real-world financial risk prediction






