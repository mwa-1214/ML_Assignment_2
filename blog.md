# Electrical Fault Detection Using Logistic Regression and Support Vector Machines

## 1. Introduction

Electrical fault detection is an important task in power systems because undetected faults can damage equipment, interrupt power supply, and create safety risks. A reliable fault detection model should identify fault conditions while minimizing missed faults and false alarms.

In this project, machine learning models were developed to classify electrical measurements as either fault or no fault. The dataset contains three-phase current measurements (`Ia`, `Ib`, `Ic`) and three-phase voltage measurements (`Va`, `Vb`, `Vc`). The target variable is `fault`, where 0 represents no fault and 1 represents a fault.

The objective was to compare Logistic Regression and Support Vector Machines (SVM) and determine which model performed best on the available dataset.

## 2. Dataset Preparation

The original dataset contained 12,001 rows. After removing two completely empty columns and renaming the target column, seven columns remained: six electrical features and the fault label.

The dataset was divided randomly into three subsets using a fixed random seed. The training set contained 7,200 rows, the validation set contained 2,400 rows, and the test set contained 2,401 rows.

Standardization was performed using statistics calculated from the training set only. The same means and standard deviations were applied to the validation and test sets. This prevented the validation and test data from influencing the scaling parameters.

Two feature sets were prepared:

- **Set R:** The six original current and voltage features.
- **Set Q:** The six original features together with their six squared terms.

Set Q was designed to help Logistic Regression capture nonlinear relationships between electrical measurements and fault conditions.

## 3. Exploratory Data Analysis

Three plots were generated to examine the dataset before model training.

### 3.1 Current Signals

![Current signals](results/plot1_currents.png)

The current plot displays the three-phase current measurements across consecutive samples, with fault conditions highlighted. It helps visualize variations in current and the behavior of the signals under different conditions.

### 3.2 Relationship Between Phase Currents

![Current scatter plot](results/plot2_scatter.png)

The scatter plot compares `Ia` and `Ib` for the two classes. It provides a visual comparison of the distributions of fault and no-fault samples. Such plots help assess whether the original features may separate the classes clearly.

### 3.3 Fault Distribution

![Fault distribution](results/plot3_fault_distribution.png)

The histogram compares the distribution of `Ia` for fault and no-fault samples. The distributions help illustrate why a model may need to consider multiple electrical features rather than relying on one current measurement alone.

## 4. Logistic Regression

Logistic Regression was implemented using the sigmoid function, a cost function, and batch gradient descent. Predictions were produced using a probability threshold of 0.5.

The learning rates 0.01, 0.1, 1, and 10 were tested over 2,000 iterations. For Set R, learning rates 0.01, 0.1, and 1 showed decreasing cost at every iteration, while learning rate 10 did not. Therefore, learning rate 1 was selected for Set R.

For Set Q, all four tested learning rates showed decreasing cost at every iteration. Learning rate 10 was selected because it produced the lowest final cost among the tested configurations.

The final Logistic Regression results were:

| Feature set | Learning rate | Final cost | Validation accuracy | Test accuracy |
|---|---:|---:|---:|---:|
| Set R | 1 | 0.677161 | 73.08% | 72.64% |
| Set Q | 10 | 0.053435 | 97.83% | 97.88% |

The results show a substantial improvement when squared features were added. This indicates that the additional features helped the model represent relationships that were not captured effectively by the original six features alone.

## 5. Support Vector Machines

SVM models were trained using Set R, as required by the assignment. Four values of `C` were tested for the linear kernel, and eight combinations of `C` and `gamma` were tested for the RBF kernel.

For the linear SVM, the validation accuracy was 72.58% for all four tested values of `C`. The selected setting was `C = 0.1`, following the first-highest-result selection in the implementation.

For the RBF SVM, the highest validation accuracy was achieved with `C = 100` and `gamma = 1`. Its validation accuracy was 99.67%.

The best SVM configurations were:

| Kernel | C | Gamma | Validation accuracy | Support vectors |
|---|---:|---:|---:|---:|
| Linear | 0.1 | — | 72.58% | 5,847 |
| RBF | 100 | 1 | 99.67% | 274 |

The linear SVM performed similarly to Logistic Regression on Set R because both models use the original six features and learn linear decision boundaries. The RBF kernel can represent more complex boundaries, which appears to be useful for this dataset.

## 6. Final Test Results

After model selection using the training and validation sets, all four final models were evaluated on the held-out test set.

| Model | Test accuracy | True negatives | False alarms | Missed faults | True positives |
|---|---:|---:|---:|---:|---:|
| Logistic Regression, Set R | 72.64% | 1,305 | 0 | 657 | 439 |
| Logistic Regression, Set Q | 97.88% | 1,299 | 6 | 45 | 1,051 |
| Linear SVM | 73.59% | 1,305 | 0 | 634 | 462 |
| RBF SVM | 99.58% | 1,301 | 4 | 6 | 1,090 |

The RBF SVM achieved the highest test accuracy at 99.58%. It detected 1,090 actual fault samples and missed only 6. Logistic Regression on Set Q also performed well, achieving 97.88% test accuracy, but it missed 45 faults.

The confusion matrices demonstrate why accuracy alone is not sufficient for electrical fault detection. A model may have reasonable overall accuracy while still missing a large number of actual faults.

## 7. Discussion

### Why did Set Q improve Logistic Regression?

The original Set R features did not capture all fault patterns effectively using a linear boundary. Adding squared terms in Set Q allowed the model to represent additional nonlinear relationships. The improvement from 72.64% to 97.88% test accuracy supports the usefulness of these features for the current dataset.

### Why did the RBF SVM perform best?

The RBF kernel can model nonlinear decision boundaries without explicitly adding squared features to the input. The selected RBF SVM achieved 99.67% validation accuracy and 99.58% test accuracy. It also used fewer support vectors than the best linear SVM: 274 compared with 5,847.

### What do the missed faults mean?

A missed fault occurs when the actual label is 1 but the model predicts 0. This is particularly important in electrical protection because a missed fault may leave a dangerous condition undetected. The RBF SVM missed 6 faults, while Logistic Regression on Set Q missed 45. These results make the RBF SVM the strongest model among those evaluated in this experiment.

### Limitations

The data was randomly split even though the rows represent consecutive time samples. Neighboring samples may be similar, and similar samples can appear in both training and test sets. As a result, the test accuracy may be optimistic compared with performance on independent data collected under different operating conditions.

The results should therefore be validated using independent measurements and additional fault conditions before practical deployment.

## 8. Conclusion

This project compared Logistic Regression on two feature sets and two SVM kernels for electrical fault detection. Set Q improved Logistic Regression substantially by adding squared features, while the RBF SVM achieved the best overall test performance.

The RBF SVM reached 99.58% test accuracy, with 4 false alarms and 6 missed faults. Based on the recorded results, it is the preferred model among the four evaluated configurations. However, practical deployment would require further validation, particularly because missed faults can have serious consequences.

The project demonstrates the importance of feature engineering, validation-based model selection, and evaluating classification errors in addition to overall accuracy.