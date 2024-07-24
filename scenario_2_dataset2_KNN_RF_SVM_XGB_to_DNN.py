import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.combine import SMOTETomek
from joblib import dump, load
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import logging
import seaborn as sns

# Function to create directories
def create_directories(directory_path):
    os.makedirs(directory_path, exist_ok=True)
    return directory_path

# Function to load and clean data
def load_clean_data(data_path):
    logging.debug("Loading and cleaning data.")
    bank = pd.read_csv(data_path)
    
    # Drop unnecessary columns
    bank = bank.drop(bank.columns[21:24], axis=1)
    
    # Check for missing data
    print("Shape of dataset:", bank.shape)
    bank.dropna(inplace=True)
    print("Shape after dropping NA:", bank.shape)
    
    # Create dummy variable for Customer Attrition
    bank2 = bank.copy()
    att = pd.get_dummies(bank2['Attrition_Flag'], drop_first=True)
    bank2 = bank2.drop('Attrition_Flag', axis=1)
    bank2 = bank2.join(att)
    
    # Encode categorical variables
    categorical_columns = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    bank2 = pd.get_dummies(bank2, columns=categorical_columns, drop_first=True)
    
    # Check the column names to ensure the correct column is used
    print("Columns after creating dummy variables:", bank2.columns)
    
    # Extract features and target
    X = bank2.drop(['Existing Customer'], axis=1)
    y = bank2['Existing Customer'].astype(int)
    
    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Combine X and y into a single DataFrame and save to CSV
    combined_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    combined_df['Churn'] = y
    
    cleaned_data_path = os.path.join(os.path.dirname(data_path), 'cleaned_data')
    os.makedirs(cleaned_data_path, exist_ok=True)
    
    combined_df.to_csv(os.path.join(cleaned_data_path, 'cleaned_data.csv'), index=False)
    
    logging.debug(f"Cleaned data saved to {cleaned_data_path}")
    
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=42)
    testX, valX, testy, valy = train_test_split(testX, testy, test_size=0.5, random_state=42)
    
    logging.debug(f"trainX shape: {trainX.shape}, trainy shape: {trainy.shape}")
    logging.debug(f"testX shape: {testX.shape}, testy shape: {testy.shape}")
    logging.debug(f"valX shape: {valX.shape}, valy shape: {valy.shape}")

    shapes_info = {
        'X shape': str(X.shape),
        'y shape': str(y.shape),
        'trainX shape': str(trainX.shape),
        'testX shape': str(testX.shape),
        'trainy shape': str(trainy.shape),
        'testy shape': str(testy.shape),
        'valX shape': str(valX.shape),
        'valy shape': str(valy.shape)
    }
    
    return bank, bank2, X, y, trainX, testX, trainy, testy, valX, valy, shapes_info

# Function to plot ROC curve
def plot_roc_curve(fpr, tpr, roc_auc, name, folder):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.xticks(rotation=45)  # Rotate x-axis labels 45 degrees
    plt.savefig(os.path.join(folder, f'roc_curve_{name}.png'))
    plt.close()

# Function to plot Precision-Recall curve
def plot_precision_recall_curve(precision, recall, average_precision, name, folder):
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, where='post', color='b', alpha=0.7, label='Precision-Recall curve (area = %0.2f)' % average_precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="upper right")
    plt.xticks(rotation=45)  # Rotate x-axis labels 45 degrees
    plt.savefig(os.path.join(folder, f'precision_recall_curve_{name}.png'))
    plt.close()

# Function to plot training history
def plot_history(history, result_folder, filename="training_history"):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig(os.path.join(result_folder, f"{filename}.png"))
    plt.close()

# Function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, name, folder):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    # plt.title(f'Confusion Matrix: {name}')
    plt.xticks(rotation=45)  # Rotate x-axis labels 45 degrees
    plt.savefig(os.path.join(folder, f'confusion_matrix_{name}.png'))
    plt.close()
    
# Function to calculate classification metrics
def calculate_classification_metrics(y_true, y_pred):
    y_true = np.array(y_true).flatten()  # Ensure y_true is 1-dimensional
    y_pred = np.array(y_pred).flatten()  # Ensure y_pred is 1-dimensional

    accuracy = np.mean(y_true == y_pred)
    error_rate = 1 - accuracy
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return accuracy, error_rate, sensitivity, specificity

# Function to evaluate a model and append its report to a DataFrame
def evaluate_model(name, model, features, true_labels, folder, all_reports, fold):
    start_time = time.time()
    y_pred = model.predict(features)
    end_time = time.time()
    train_time = end_time - start_time
    y_pred_proba = model.predict_proba(features)[:, 1] if hasattr(model, "predict_proba") else None

    report = classification_report(true_labels, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df['model'] = name
    report_df['fold'] = fold
    report_df['time'] = train_time

    all_reports = pd.concat([all_reports, report_df], axis=0)

    model_folder = create_directories(os.path.join(folder, name))
    
    if y_pred_proba is not None:
        fpr, tpr, thresholds = roc_curve(true_labels, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plot_roc_curve(fpr, tpr, roc_auc, name, model_folder)

        precision, recall, _ = precision_recall_curve(true_labels, y_pred_proba)
        auc_pr = average_precision_score(true_labels, y_pred_proba)
        plot_precision_recall_curve(precision, recall, auc_pr, name, model_folder)

        accuracy, error_rate, sensitivity, specificity = calculate_classification_metrics(true_labels, y_pred)

        metrics = {
            'ROC-AUC': roc_auc,
            'AUC-PR': auc_pr,
            'Accuracy': accuracy,
            'Error Rate': error_rate,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Training Time': train_time
        }
    else:
        metrics = {'Training Time': train_time}

    # Plot the confusion matrix
    plot_confusion_matrix(true_labels, y_pred, name, model_folder)
    
    return metrics, all_reports

# Function to create base models
def create_base_models(best_params):
    models = [
        ('KNN', KNeighborsClassifier()),
        ('RF', RandomForestClassifier(**best_params['RF'], random_state=42)),
        ('SVM', SVC(probability=True)),
        ('XGB', XGBClassifier(**best_params['XGB'], random_state=42))
    ]
    return models

# Function to fit a stacked model using DNN
def fit_stacked_model(inputX, inputy, result_folder):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=inputX.shape[1]))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(inputX, inputy, epochs=150, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])
    plot_history(history, result_folder, "my_model_training")
    return model

# Function to create a stacked dataset
def stacked_dataset(members, inputX):
    stackX = None
    for name, model in members:
        yhat = model.predict_proba(inputX)[:, 1]
        stackX = np.column_stack([stackX, yhat]) if stackX is not None else yhat.reshape(-1, 1)
    return stackX

# Function for stacked prediction
def stacked_prediction(members, stacked_model, inputX):
    stackedX = stacked_dataset(members, inputX)
    yhat_proba = stacked_model.predict(stackedX)
    yhat = (yhat_proba > 0.5).astype(int)
    return yhat, yhat_proba

# Function to plot comparison of metrics across models
def plot_metric_comparison(metrics_dict, metric_name, result_folder):
    logging.debug(f"Plotting comparison for {metric_name}.")
    fig, ax = plt.subplots()
    model_names = list(metrics_dict.keys())
    metric_values = [metrics[metric_name] for metrics in metrics_dict.values()]
    
    ax.bar(model_names, metric_values, color=['gold', 'lightgreen', 'lightcoral', 'lightskyblue', 'lightpink', 'lightyellow', 'lightgray'])
    for index, value in enumerate(metric_values):
        ax.text(index, value, f'{value:.4f}', ha='center', va='bottom')
    
    ax.set_ylabel(metric_name)
    plt.xticks(rotation=45)  # Rotate x-axis labels 45 degrees
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, f'{metric_name}_comparison.png'))
    plt.close(fig)
    logging.debug(f"{metric_name} comparison plot saved: {result_folder}/{metric_name}_comparison.png")

# Function to plot a bar chart of model metrics
def plot_metrics_bar(metrics_dict, result_folder):
    for model, metrics in metrics_dict.items():
        logging.debug(f"Plotting model metrics bar chart for {model}.")
        fig, ax = plt.subplots()
        metric_names = ['Accuracy', 'F1-score', 'Precision', 'ROC-AUC', 'AUC-PR']
        metric_values = [metrics[name] for name in metric_names]
        
        ax.bar(metric_names, metric_values, color=['gold', 'lightgreen', 'lightcoral', 'lightskyblue', 'lightpink', 'lightyellow', 'lightgray'])
        for index, value in enumerate(metric_values):
            ax.text(index, value, f'{value:.4f}', ha='center', va='bottom')
        
        # plt.title(f'Metrics for {model}')
        plt.xticks(rotation=45)  # Rotate x-axis labels 45 degrees
        plt.tight_layout()
        model_folder = create_directories(os.path.join(result_folder, model))
        plt.savefig(os.path.join(model_folder, f'{model}_metrics.png'))
        plt.close(fig)
        logging.debug(f"Model metrics bar plot saved: {model_folder}/{model}_metrics.png")

# Main function
def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.debug("Starting main function.")
    
    directory_work = os.getcwd()
    experiment_folder = '/data2/cmdir/home/anhnv/vvhieu/CIT/dataset2_BankChurners'
    model_name = "scenario_2_dataset2_KNN_RF_SVM_XGB_to_DNN"
    result_folder = create_directories(os.path.join(experiment_folder, model_name))

    data_path = os.path.join(experiment_folder, 'BankChurners.csv')
    # X, y = load_clean_data(data_path)
    bank, bank2, X, y, trainX, testX, trainy, testy, valX, valy, shapes_info = load_clean_data(data_path)

    logging.debug(f"Initial X shape: {X.shape}")
    logging.debug(f"Initial y shape: {y.shape}")

    # Split the original data into training and test sets for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTETomek
    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X, y)

    # Split the resampled data into training and test sets for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Grid search for hyperparameters
    model_params = {
        'RF': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
        'XGB': {'n_estimators': [100, 200], 'max_depth': [10, 20], 'learning_rate': [0.01, 0.1]}
    }

    best_params = {}
    rf = RandomForestClassifier(random_state=42)
    xgb = XGBClassifier(random_state=42)
    
    # Grid search for RandomForest
    grid_rf = GridSearchCV(rf, model_params['RF'], cv=5, scoring='accuracy')
    grid_rf.fit(X, y)
    best_params['RF'] = grid_rf.best_params_

    # Grid search for XGBClassifier
    grid_xgb = GridSearchCV(xgb, model_params['XGB'], cv=5, scoring='accuracy')
    grid_xgb.fit(X, y)
    best_params['XGB'] = grid_xgb.best_params_

    base_models = create_base_models(best_params)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_classification_reports = pd.DataFrame()

    oof_preds = np.zeros((X.shape[0], len(base_models)))
    all_model_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        fold_trainX, fold_valX = X[train_idx], X[val_idx]
        fold_trainy, fold_valy = y[train_idx], y[val_idx]
        fold_trainX, fold_trainy = SMOTETomek(random_state=42).fit_resample(fold_trainX, fold_trainy)

        for i, (name, model) in enumerate(base_models):
            model.fit(fold_trainX, fold_trainy)
            preds = model.predict_proba(fold_valX)[:, 1]
            oof_preds[val_idx, i] = preds
            
            model_folder = create_directories(os.path.join(result_folder, name))
            dump(model, os.path.join(model_folder, f'{name}_fold_{fold}_model.joblib'))

            metrics, all_classification_reports = evaluate_model(name, model, fold_valX, fold_valy, model_folder, all_classification_reports, fold)
            metrics['model'] = name
            metrics['Precision'] = round(precision_score(fold_valy, model.predict(fold_valX)), 3)
            metrics['Recall'] = round(recall_score(fold_valy, model.predict(fold_valX)), 3)
            metrics['F1-score'] = round(f1_score(fold_valy, model.predict(fold_valX)), 3)
            metrics['Accuracy'] = round(accuracy_score(fold_valy, model.predict(fold_valX)), 3)
            all_model_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_model_metrics)
    metrics_filename = os.path.join(result_folder, 'base_model_metrics.csv')
    metrics_df.to_csv(metrics_filename, index=False)
    print(f"Saved model metrics to {metrics_filename}")

    print("Shape of oof_preds:", oof_preds.shape)

    # Define start time for stacked model training
    start_time = time.time()

    stacked_model_folder = create_directories(os.path.join(result_folder, 'Scenario 2'))
    stacked_model = fit_stacked_model(oof_preds, y, stacked_model_folder)

    stacked_test = np.zeros((X_test.shape[0], len(base_models)))
    for i, (_, model) in enumerate(base_models):
        model_folder = create_directories(os.path.join(result_folder, base_models[i][0]))
        model = load(os.path.join(model_folder, f'{base_models[i][0]}_fold_{kf.n_splits-1}_model.joblib'))
        stacked_test[:, i] = model.predict_proba(X_test)[:, 1]

    stacked_pred, stacked_pred_proba = stacked_prediction(base_models, stacked_model, X_test)
    report = classification_report(y_test, stacked_pred, output_dict=True)

    fpr, tpr, thresholds = roc_curve(y_test, stacked_pred_proba)
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc, 'Scenario 2', stacked_model_folder)

    precision, recall, _ = precision_recall_curve(y_test, stacked_pred_proba)
    auc_pr = average_precision_score(y_test, stacked_pred_proba)
    plot_precision_recall_curve(precision, recall, auc_pr, 'Scenario 2', stacked_model_folder)

    accuracy, error_rate, sensitivity, specificity = calculate_classification_metrics(y_test, stacked_pred)
    accuracy = round(accuracy_score(y_test, stacked_pred), 3)
    
    report_df = pd.DataFrame(report).transpose()
    report_df['model'] = 'Scenario 2'
    report_df['Precision'] = round(precision_score(y_test, stacked_pred), 3)
    report_df['Recall'] = round(recall_score(y_test, stacked_pred), 3)
    report_df['F1-score'] = round(f1_score(y_test, stacked_pred), 3)
    report_df['Accuracy'] = round(accuracy_score(y_test, stacked_pred), 3)
    
    report_df['ROC-AUC'] = roc_auc
    report_df['AUC-PR'] = auc_pr
    report_df['accuracy'] = accuracy
    report_df['error_rate'] = error_rate
    report_df['sensitivity'] = sensitivity
    report_df['specificity'] = specificity
    report_df['train_time'] = time.time() - start_time

    all_classification_reports = pd.concat([all_classification_reports, report_df], axis=0)
    all_classification_reports.to_csv(os.path.join(result_folder, 'all_classification_reports.csv'), index=True)

    # Add Scenario 2 metrics to the metrics dictionary
    stacked_metrics = {
        'Accuracy': accuracy,
        'F1-score': report_df.loc['weighted avg', 'f1-score'],
        'Precision': report_df.loc['weighted avg', 'precision'],
        'ROC-AUC': roc_auc,
        'AUC-PR': auc_pr,
        'Training Time': report_df['train_time'].iloc[0]
    }

    # Combine metrics from all folds and include the stacked model metrics
    metrics_dict = metrics_df.groupby('model').mean().to_dict('index')
    metrics_dict['Scenario 2'] = stacked_metrics

    # Plot metric comparison
    plot_metric_comparison(metrics_dict, 'Accuracy', result_folder)
    plot_metric_comparison(metrics_dict, 'F1-score', result_folder)
    plot_metric_comparison(metrics_dict, 'Precision', result_folder)
    plot_metric_comparison(metrics_dict, 'ROC-AUC', result_folder)
    plot_metric_comparison(metrics_dict, 'AUC-PR', result_folder)

    # Plot metrics bar chart for all models
    plot_metrics_bar(metrics_dict, result_folder)

if __name__ == "__main__":
    main()
