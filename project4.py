import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore') 
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, make_scorer
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#load dataset
file_path = 'DataSet.csv'
df = pd.read_csv(file_path)

#count samples and features
num_samples, num_features = df.shape
print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")
print("\n" + "=" * 50 + "\n")
#Column information
print("column information: ")
print(df.info())
print("\n" + "=" * 50 + "\n")
#number of missing values
print("number of missing values in each column")
missing_counts = df.isnull().sum()
print(missing_counts[missing_counts > 0].sort_values(ascending=False))
if missing_counts.sum() == 0:
    print("there is no missing values")
print("\n" + "=" * 50 + "\n")
#class ratio 
if 'fraudulent' in df.columns:
    print("class ratio in fraudulent: ")
    class_counts = df['fraudulent'].value_counts()
    class_percentages = df['fraudulent'].value_counts(normalize=True) * 100

    class_report = pd.DataFrame({
        'number': class_counts,
        'Percentage': class_percentages.round(2).astype(str) + '%'
    })
    print(class_report)
else:
    print("column fraudulent not found")

# Remove columns with too many missing values
columns_to_drop = ['salary_range', 'department', 'in_balanced_dataset']
df = df.drop(columns=columns_to_drop, axis=1)

# Replace missing values in text columns with an empty string
text_columns = ['location', 'company_profile', 'requirements', 'benefits', 
                'employment_type', 'required_experience', 'required_education', 
                'industry', 'function']
df[text_columns] = df[text_columns].fillna('')

# Combine main text columns into a single column 
df['full_text'] = df['title'] + ' ' + df['company_profile'] + ' ' + \
                  df['description'] + ' ' + df['requirements'] + ' ' + \
                  df['benefits']
print("-" * 50)
print(f"remaining column: {df.shape[1]}")
print("-" * 50)


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    #tokenizing
    tokens = word_tokenize(text)
    
    #lemmitizing
    processed_tokens = []
    for word in tokens:
        if word not in stop_words:
            processed_tokens.append(lemmatizer.lemmatize(word))
            
    #Recombine tokens into a string
    return " ".join(processed_tokens)

#Applying a preprocessing function to a text column
df['processed_text'] = df['full_text'].apply(preprocess_text)

#TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=20000) 


tfidf_matrix = tfidf.fit_transform(df['processed_text'])

# Display the dimensions of the final vector
print(f"TF-IDF matrix dimensions: {tfidf_matrix.shape}")
print(f" number of samples: {tfidf_matrix.shape[0]} number of features: {tfidf_matrix.shape[1]})")
print("-" * 50)

# Convert 'f' to 0 (legal) and 't' to 1 (fraud)
y = df['fraudulent'].map({'f': 0, 't': 1})

X = tfidf_matrix

#2. Stratified Separation
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,  
    stratify=y        
)

# Check the dimensions of the new dataset
print(f"X_train dimensions:{X_train.shape}")
print(f"X_test dimensions:{X_test.shape}")

# Checking the ratio of classes in the training set
print("\n Distribution of classes in the training set (y_train):")
print(y_train.value_counts(normalize=True) * 100)

# Check the proportion of classes in the test set
print("\n Distribution of classes in the test set (y_test):")
print(y_test.value_counts(normalize=True) * 100)




print(" Starting to train a logistic regression model with class weights ")
start_time = time.time()

model_logreg = LogisticRegression(
    solver='liblinear',
    C=1.0, 
    random_state=42,
    class_weight='balanced' 
)

#model training
model_logreg.fit(X_train, y_train)

end_time = time.time()
print(f"time {end_time - start_time:.2f} seconds")
print("-" * 50)

# Prediction on test data
y_pred = model_logreg.predict(X_test)

# Model Performance Evaluation
print("Model Performance Evaluation Report")
print(classification_report(y_test, y_pred))




def train_and_evaluate_svm(kernel_type):
    print(f"\n Starting SVM training with kernel: {kernel_type.upper()} ")
    start_time = time.time()

    # Defining the model with a specific kernel and weighting the classes
    model_svm = SVC(
        kernel=kernel_type,
        C=1.0,
        random_state=42,
        class_weight='balanced'
    )

    #model training
    model_svm.fit(X_train, y_train)
    
    end_time = time.time()
    train_time = end_time - start_time
    print(f" time: {train_time:.2f} seconds")
    
    #prediction
    y_pred = model_svm.predict(X_test)

    # Performance evaluation
    print(f"\n SVM Performance Report({kernel_type.upper()})")
    print(classification_report(y_test, y_pred))
    
    f1_class_1 = f1_score(y_test, y_pred, pos_label=1)
    return f1_class_1, train_time, model_svm

# (Linear Kernel)
f1_linear, time_linear, model_svm_linear = train_and_evaluate_svm('linear')

# RBF (Radial Basis Function Kernel) (non-linear kernel)
f1_rbf, time_rbf, model_svm_rbf = train_and_evaluate_svm('rbf')


# Assuming X_train, X_test, y_train, y_test are available from previous steps.

print(" Starting KNN Hyperparameter Optimization (Grid Search) ")
start_time = time.time()

param_grid = {
    'n_neighbors': list(range(1, 16, 2)), 
    'weights': ['uniform', 'distance']
}

f1_scorer = make_scorer(f1_score, pos_label=1)

# Define the base KNN model 
knn = KNeighborsClassifier()

# Execute Grid Search 

grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    scoring=f1_scorer,
    cv=5, 
    verbose=1,
    n_jobs=-1 
)

# Fit the grid search on the training data
grid_search.fit(X_train, y_train)

end_time = time.time()
print(f" Optimization completed successfully. Total time: {end_time - start_time:.2f} seconds")
print("-" * 50)

# Report Grid Search Results 
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f" Best hyperparameters found: {best_params}")
print(f" Best F1-Score on cross-validation: {best_score:.4f}")
print("-" * 50)

# Evaluate the final best model on the test set 
best_knn_model = grid_search.best_estimator_
y_pred_knn = best_knn_model.predict(X_test)

print(" Optimized KNN Model Performance Report on Test Data ")
print(classification_report(y_test, y_pred_knn))


y_pred_logreg = model_logreg.predict(X_test)
y_pred_svm_linear = model_svm_linear.predict(X_test)
y_pred_svm_rbf = model_svm_rbf.predict(X_test)
y_pred_knn = best_knn_model.predict(X_test)


class_names = ['(0)', '(1)']
model_list = {
    "logistic regression": y_pred_logreg,
    "SVM linear": y_pred_svm_linear,
    "SVM non-linear(RBF)": y_pred_svm_rbf,
    "Optimized KNN": y_pred_knn
}

fig, axes = plt.subplots(1, 4, figsize=(24, 6)) 
axes = axes.flatten()
i = 0

for name, y_pred in model_list.items():
    cm = confusion_matrix(y_test, y_pred)
    
 
    FN = cm[1, 0] 
    FP = cm[0, 1] 
    

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
 
    disp.plot(ax=axes[i], cmap=plt.cm.Blues, values_format='d', colorbar=False) 
    

    axes[i].title.set_text(f"{name}\nFN: {FN}, FP: {FP}")
    
    i += 1


plt.tight_layout(rect=[0, 0.03, 1, 0.9])
plt.show()



cm_svm = np.array([[3388, 15], [27, 146]])


cm_knn = np.array([[3380, 23], [30, 143]])


def calculate_metrics(cm, model_name):
    TN, FP, FN, TP = cm.ravel()
    
    N = TN + FP + FN + TP
    P = TP + FN 
    N_true = TN + FP 
    
  
    accuracy = (TP + TN) / N
    error_rate = 1 - accuracy

    sensitivity = TP / P
    
    specificity = TN / N_true

    precision = TP / (TP + FP)
    
    f_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    
    return {
        'Model': model_name,
        'Accuracy': round(accuracy, 4),
        'Error Rate': round(error_rate, 4),
        'Sensitivity (Recall)': round(sensitivity, 4),
        'Specificity': round(specificity, 4),
        'Precision': round(precision, 4),
        'F-score (F1)': round(f_score, 4),
        'FN': FN,
        'FP': FP
    }


results_svm = calculate_metrics(cm_svm, 'SVM Linear')
results_knn = calculate_metrics(cm_knn, 'KNN Optimized')


df_results = pd.DataFrame([results_svm, results_knn])


print("Comparison of Performance Metrics:")
print(df_results.to_markdown(index=False))

