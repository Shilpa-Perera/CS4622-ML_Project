# %%
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# %%
train_path = 'speech-based-classification-layer-10/train.csv'
valid_path = 'speech-based-classification-layer-10/valid.csv'
test_path = 'speech-based-classification-layer-10/test.csv'
train = pd.read_csv(train_path)
valid = pd.read_csv(valid_path)
test = pd.read_csv(test_path)
original_train = train.copy()
original_valid = train.copy()
original_test = test.copy()

# %% [markdown]
# # Label Data

# %%
train_features = train.iloc[:, :768]
train_label_4 = train.iloc[:, 771]

valid_features = valid.iloc[:, :768]
valid_label_4 = valid.iloc[:, 771]

test_features = test.iloc[:, 1:]

# %% [markdown]
# Class Distribution Plot

# %%
class_counts = train_label_4.value_counts()
plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar', color='skyblue')

plt.xlabel('Class Label')
plt.ylabel('Count')
plt.title('Class Distribution')


plt.show()

# %% [markdown]
# From SVM and Random Forest classifiers choosing the best classification model
# 1. Stratified K Fold model is used for cross validation
# 2. Default parameters of classifcation models are used
# 3. Since classes are balanced accuracy score is used for the evaluation

# %%
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



classifiers = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
}


best_model = None
best_accuracy = 0.0


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits)

for name, clf in classifiers.items():
    total_score = 0.0

    for train_index, test_index in skf.split(train_features, train_label_4):
        X_train, X_test = train_features.iloc[train_index], train_features.iloc[test_index]
        y_train, y_test = train_label_4[train_index], train_label_4[test_index]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1_score_ = f1_score(y_test, y_pred , average = 'weighted')
        total_score += f1_score_

    average_score = total_score / n_splits
    print(f"{name} - Average F1 Score: {average_score}")


    if average_score > best_accuracy:
        best_model = clf
        best_accuracy = average_score

print(f"Best Model: {type(best_model).__name__}")

# %%
from sklearn.preprocessing import RobustScaler
transformer = RobustScaler()
scaled_train_features = transformer.fit_transform(train_features)
scaled_valid_features = transformer.fit_transform(valid_features)
scaled_test_features = transformer.fit_transform(test_features)

# %% [markdown]
# PCA Transformation 

# %%
from sklearn.decomposition import PCA

def performPca(train_input, valid_input,test_input, n_components):
    pca = PCA(n_components=n_components)
    train_reduced = pca.fit_transform(train_input)
    valid_reduced = pca.transform(valid_input)
    test_reduced = pca.transform(test_input)
    train_reduced_df = pd.DataFrame(train_reduced, columns=[f"new_feature_{i+1}" for i in range(train_reduced.shape[1])])
    valid_reduced_df = pd.DataFrame(valid_reduced, columns=[f"new_feature_{i+1}" for i in range(valid_reduced.shape[1])])
    test_reduced_df = pd.DataFrame(test_reduced, columns=[f"new_feature_{i+1}" for i in range(test_reduced.shape[1])])


    return train_reduced_df, valid_reduced_df,test_reduced_df

# %%
train_reduced_df, valid_reduced_df,test_reduced_df = performPca(scaled_train_features, scaled_valid_features, scaled_test_features, 0.99)

# %%
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

model = SVC()
model.fit(train_reduced_df, train_label_4)


# %%
predictions = model.predict(valid_reduced_df)
print(f"Accuracy: {f1_score(valid_label_4, predictions , average='weighted')}")

# %% [markdown]
# Addressing Class Imbalance Issues
# 1. Use SMOTE oversampling to address class imbalance issue

# %%
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='auto', random_state=42)
train_feature_resampled, train_label_4_resampled = smote.fit_resample(train_reduced_df, train_label_4)

# %% [markdown]
# Accuracy after oversampling

# %%
model_ = SVC()
model_.fit(train_feature_resampled, train_label_4_resampled)
y_pred = model_.predict(valid_reduced_df)


# %%
from sklearn.metrics import accuracy_score, f1_score
accuracy = f1_score(valid_label_4, y_pred, average='weighted')
print(f"F1 Score: {accuracy}")

# %%
#0.9784248577478692
test_predictions = model_.predict(test_reduced_df)

# %%
#Write to file
test_pred_df = pd.DataFrame(test_predictions, columns=['label_4'])
test_pred_df.to_csv('predictions/label_4.csv', index=False)

# %% [markdown]
# # Hyper parameter tuning

# %%
fin_model_ = SVC(C=0.1, gamma=0.01, kernel='rbf')
fin_model_.fit(train_reduced_df, train_label_4_resampled)
fin_y_pred = fin_model_.predict(valid_reduced_df)

# %%
from sklearn.metrics import accuracy_score, f1_score
accuracy = f1_score(valid_label_4, fin_y_pred, average='weighted')
print(f"F1 Score: {accuracy}")

# %%
test_predictions = fin_model_.predict(test_reduced_df)

# %%
#Write to file
test_pred_df = pd.DataFrame(test_predictions, columns=['label_4'])
test_pred_df.to_csv('predictions/label_4.csv', index=False)

# %% [markdown]
# ## Hyperparameter Tuning

# %%
from sklearn.model_selection import GridSearchCV
  
# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100], 
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['rbf' , 'linear' , 'poly']} 
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
  
# fitting the model for grid search
grid.fit(train_reduced_df, train_label_4_resampled)

# %%
fin_model_ = SVC(C=0.1 , gamma=1 , kernel='linear')
fin_model_.fit(train_reduced_df, train_label_4_resampled)
y_pred_fin = fin_model_.predict(valid_reduced_df)

# %%
accuracy = f1_score(valid_label_4, y_pred, average='weighted')
print(f"F1 Score: {accuracy}")

# %%
test_predictions = fin_model_.predict(test_reduced_df)

# %%
#Write to file
test_pred_df = pd.DataFrame(test_predictions, columns=['label_4'])
test_pred_df.to_csv('predictions/label_4.csv', index=False)

# %% [markdown]
# Final Submission

# %%
import pandas as pd


file_paths = ['predictions/label_1.csv', 'predictions/label_2.csv', 'predictions/label_3.csv', 'predictions/label_4.csv']

# Initialize an empty DataFrame
merged_data = pd.DataFrame()

# Initialize a counter for the "ID" column
id_counter = 1

# Iterate through the file paths and merge the columns
for i, file_path in enumerate(file_paths, start=1):
    # Read each CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Extract the label name from the first row
    label_name = df.iloc[0, 0]
    
    # Rename the single column to the corresponding label
    df = df.rename(columns={df.columns[0]: f'label_{i}'})
    
    # Add the labeled column to the merged DataFrame
    merged_data = pd.concat([merged_data, df], axis=1)

# Create the "ID" column with sequential numbers
id_column = pd.Series(range(1, merged_data.shape[0] + 1), name="ID")

# Insert the "ID" column as the first column
merged_data.insert(0, "ID", id_column)

# Save the merged data to a new CSV file
merged_data.to_csv('combined_data.csv', index=False)


