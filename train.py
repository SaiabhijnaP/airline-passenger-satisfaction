"""
Model Training, Evaluation, and Kaggle Submission Generation
Project: Airline Passenger Satisfaction Prediction
Dataset: Kaggle Playground Series - Season 6, Episode 10
Evaluation Metric: Area Under the ROC Curve (ROC AUC)
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Matplotlib configuration
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')


def train_and_evaluate(
    train_path='train.csv',
    test_path='test.csv',
    sample_sub_path='sample_submission.csv',
    models_dir='models',
    output_dir='outputs',
    random_state=42
):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("AIRLINE PASSENGER SATISFACTION: MODEL TRAINING & BENCHMARKING PIPELINE")
    print("=" * 80)

    # 1. Load Data
    print(f"\n[1/8] Loading training dataset from {train_path}...")
    t0 = time.time()
    train_df = pd.read_csv(train_path)
    print(f"Loaded {len(train_df):,} rows x {train_df.shape[1]} columns in {time.time()-t0:.2f}s")

    # 2. Features and Target Definition
    target_col = 'satisfaction'
    id_col = 'id'

    X = train_df.drop(columns=[id_col, target_col])
    # Convert boolean target to integer (0: Dissatisfied, 1: Satisfied)
    y = train_df[target_col].astype(int)

    categorical_features = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    numerical_features = [c for c in X.columns if c not in categorical_features]

    print(f"\n[2/8] Feature Separation:")
    print(f"  - Target Variable: '{target_col}' (Binary: 0=Dissatisfied, 1=Satisfied)")
    print(f"  - Categorical Features ({len(categorical_features)}): {categorical_features}")
    print(f"  - Numerical Features ({len(numerical_features)}): {numerical_features}")

    # 3. Stratified Train-Validation Split (No Data Leakage)
    print(f"\n[3/8] Splitting labeled data (80% Train, 20% Validation) with stratification...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    print(f"  - Training Set: {len(X_train):,} samples (Satisfied: {y_train.mean()*100:.2f}%)")
    print(f"  - Validation Set: {len(X_val):,} samples (Satisfied: {y_val.mean()*100:.2f}%)")

    # 4. Construct Preprocessing Pipeline with ColumnTransformer
    print("\n[4/8] Building Scikit-Learn Preprocessing Pipeline...")
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_features),
            ('cat', cat_pipeline, categorical_features)
        ],
        remainder='drop'
    )

    # 5. Define Candidate Classification Models
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=random_state, solver='lbfgs'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=18, min_samples_split=10,
            n_jobs=-1, random_state=random_state
        ),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.1, max_depth=12,
            min_samples_leaf=20, random_state=random_state
        )
    }

    results = []
    trained_pipelines = {}
    val_preds_dict = {}
    val_probs_dict = {}

    print("\n[5/8] Training and Evaluating Candidate Models...")
    print("-" * 80)
    print(f"{'Model Name':<25} | {'Train Time':<10} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'ROC-AUC':<9}")
    print("-" * 80)

    for name, clf in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        start_train = time.time()
        pipeline.fit(X_train, y_train)
        train_duration = time.time() - start_train

        val_preds = pipeline.predict(X_val)
        val_probs = pipeline.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, val_preds)
        prec = precision_score(y_val, val_preds, zero_division=0)
        rec = recall_score(y_val, val_preds, zero_division=0)
        f1 = f1_score(y_val, val_preds, zero_division=0)
        roc_auc = roc_auc_score(y_val, val_probs)

        results.append({
            'Model': name,
            'Train_Time_Sec': round(train_duration, 2),
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1_Score': round(f1, 4),
            'ROC_AUC': round(roc_auc, 4)
        })

        trained_pipelines[name] = pipeline
        val_preds_dict[name] = val_preds
        val_probs_dict[name] = val_probs

        print(f"{name:<25} | {train_duration:<9.2f}s | {acc:<9.4f} | {prec:<9.4f} | {rec:<9.4f} | {f1:<9.4f} | {roc_auc:<9.4f}")

    print("-" * 80)

    # 6. Model Comparison and Visualizations
    print("\n[6/8] Generating Model Comparison & Confusion Matrix Visualizations...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    print("  -> Saved outputs/model_comparison.csv")

    # Plot Comparison Metrics
    fig, ax = plt.subplots(figsize=(11, 6))
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
    results_melted = pd.melt(results_df, id_vars=['Model'], value_vars=metrics_to_plot,
                             var_name='Metric', value_name='Score')

    sns.barplot(data=results_melted, x='Metric', y='Score', hue='Model',
                palette=['#3498db', '#9b59b6', '#2ecc71'], ax=ax)
    ax.set_ylim(0.80, 1.0)
    ax.set_title('Cross-Model Performance Comparison (Validation Set)', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Score (0.0 to 1.0)', fontsize=11)
    ax.set_xlabel('Evaluation Metric', fontsize=11)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=8, rotation=0, xytext=(0, 2),
                        textcoords='offset points')
    ax.legend(title='Model', loc='lower right', frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '07_model_comparison.png'), dpi=300)
    plt.close()
    print("  -> Saved outputs/07_model_comparison.png")

    # Plot Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (name, preds) in enumerate(val_preds_dict.items()):
        cm = confusion_matrix(y_val, preds)
        sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Dissatisfied (0)', 'Satisfied (1)'],
                    yticklabels=['Dissatisfied (0)', 'Satisfied (1)'],
                    cbar=False)
        axes[idx].set_title(f'{name}\nConfusion Matrix', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
        axes[idx].set_ylabel('True Label', fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '08_confusion_matrices.png'), dpi=300)
    plt.close()
    print("  -> Saved outputs/08_confusion_matrices.png")

    # Select Best Model based on Kaggle competition metric: ROC AUC
    best_row = results_df.sort_values(by=['ROC_AUC', 'F1_Score'], ascending=False).iloc[0]
    best_model_name = best_row['Model']
    best_pipeline = trained_pipelines[best_model_name]

    print(f"\n>>> Selected Best Model: '{best_model_name}'")
    print(f"    Validation ROC-AUC: {best_row['ROC_AUC']:.4f} (Kaggle S6E10 Primary Metric)")
    print(f"    Validation Accuracy: {best_row['Accuracy']:.4f}")
    print(f"    Validation F1-Score: {best_row['F1_Score']:.4f}")

    # 7. Feature Importance Computation
    print("\n[7/8] Analyzing Feature Importance for Best Model...")
    # Sample validation records for permutation importance to balance speed and statistical rigor
    sample_val_size = min(15000, len(X_val))
    val_sample_idx = np.random.RandomState(random_state).choice(len(X_val), sample_val_size, replace=False)
    X_val_sample = X_val.iloc[val_sample_idx]
    y_val_sample = y_val.iloc[val_sample_idx]

    perm_importance = permutation_importance(
        best_pipeline, X_val_sample, y_val_sample,
        n_repeats=5, random_state=random_state, scoring='roc_auc', n_jobs=-1
    )

    feat_names = list(X.columns)
    importance_df = pd.DataFrame({
        'Feature': feat_names,
        'Importance_Mean': perm_importance.importances_mean,
        'Importance_Std': perm_importance.importances_std
    }).sort_values(by='Importance_Mean', ascending=False).reset_index(drop=True)

    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    print("  -> Saved outputs/feature_importance.csv")

    fig, ax = plt.subplots(figsize=(11, 7))
    top_feats = importance_df.head(15).sort_values(by='Importance_Mean', ascending=True)
    ax.barh(top_feats['Feature'], top_feats['Importance_Mean'],
            xerr=top_feats['Importance_Std'], color='#2ecc71', edgecolor='#27ae60', alpha=0.9, capsize=4)
    ax.set_xlabel('Permutation Importance (Drop in ROC-AUC)', fontsize=11, fontweight='bold')
    ax.set_title(f'Feature Importance: Top Drivers of Passenger Satisfaction ({best_model_name})\n(Permutation on Validation Data)',
                 fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '09_feature_importance.png'), dpi=300)
    plt.close()
    print("  -> Saved outputs/09_feature_importance.png")

    print("\n  Top 5 Most Influential Features:")
    for i, row in importance_df.head(5).iterrows():
        print(f"    {i+1}. {row['Feature']}: ROC-AUC drop = {row['Importance_Mean']:.4f}")
    print("\n  * Important Note: These features demonstrate strong statistical associations with reported satisfaction,")
    print("    providing actionable operational focus. However, correlation and model feature importance do not prove direct causality.")

    # Save Best Model Pipeline and Metadata
    best_model_file = os.path.join(models_dir, 'best_model_pipeline.joblib')
    joblib.dump(best_pipeline, best_model_file)
    print(f"\n  -> Saved best pipeline to {best_model_file}")

    feature_metadata = {
        'all_features': list(X.columns),
        'categorical_features': categorical_features,
        'numerical_features': numerical_features,
        'target_col': target_col,
        'target_classes': ['Dissatisfied', 'Satisfied'],
        'best_model_name': best_model_name
    }
    with open(os.path.join(models_dir, 'feature_metadata.json'), 'w') as f:
        json.dump(feature_metadata, f, indent=4)

    model_metrics = {
        'best_model': best_model_name,
        'best_metrics': best_row.to_dict(),
        'all_models': results,
        'top_features': importance_df.head(10).to_dict(orient='records'),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(os.path.join(models_dir, 'model_metrics.json'), 'w') as f:
        json.dump(model_metrics, f, indent=4)
    print("  -> Saved models/model_metrics.json & models/feature_metadata.json")

    # 8. Generate Kaggle Official Submission File
    if os.path.exists(test_path) and os.path.exists(sample_sub_path):
        print(f"\n[8/8] Generating Kaggle Competition Submission from {test_path}...")
        t_sub = time.time()
        test_df = pd.read_csv(test_path)
        sample_sub = pd.read_csv(sample_sub_path)

        # Drop id column for model input
        X_test = test_df.drop(columns=[id_col])

        # Generate predicted probabilities for the positive class (1: Satisfied)
        test_probs = best_pipeline.predict_proba(X_test)[:, 1]

        submission = pd.DataFrame({
            'id': test_df[id_col],
            'satisfaction': test_probs
        })

        # Rigorous verification against sample_submission.csv
        assert len(submission) == len(sample_sub), f"Row count mismatch: {len(submission)} vs {len(sample_sub)}"
        assert list(submission.columns) == list(sample_sub.columns), f"Column mismatch: {list(submission.columns)} vs {list(sample_sub.columns)}"
        assert submission['id'].equals(sample_sub['id']), "ID ordering mismatch with sample submission!"
        assert submission['satisfaction'].isnull().sum() == 0, "Submission contains NaN predictions!"

        sub_file = 'submission.csv'
        submission.to_csv(sub_file, index=False)
        print(f"  -> Successfully generated {sub_file} in {time.time()-t_sub:.2f}s!")
        print(f"     Rows: {len(submission):,}, Columns: {list(submission.columns)}")
        print(f"     Preview:\n{submission.head(4)}\n...\n{submission.tail(2)}")
    else:
        print("\n[8/8] Test or sample submission file not found, skipping submission generation.")

    print("\n" + "=" * 80)
    print("MODEL PIPELINE & SUBMISSION COMPLETE!")
    print("=" * 80)
    return best_model_file, results_df


if __name__ == '__main__':
    train_and_evaluate()
