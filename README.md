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


