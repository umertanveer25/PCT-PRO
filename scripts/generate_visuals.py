import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pct_pro_engine.pct_pro_core import PCTProEngine
import os

# Set style for academic quality
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

def generate_research_plots():
    print("🚀 Generating World-Class Research Visualizations for PCT-PRO...")
    
    # 1. Load Data (Optimized for Memory)
    path = r"c:\Users\umert\Downloads\KFS_IOT\Robustness_Datasets\WUSTL-EHMS\wustl-ehms-2020 dataset.csv"
    if not os.path.exists(path):
        print(f"Error: Dataset not found at {path}")
        return

    # Use smaller subset for plotting to avoid 3TB RAM requirement
    df = pd.read_csv(path, nrows=5000)
    
    # Drop high cardinality columns that explode dimensions
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].nunique() > 50 and col != 'Label':
            df = df.drop(col, axis=1)
            
    X = pd.get_dummies(df.drop(['Label'], axis=1), drop_first=True).values
    y = LabelEncoder().fit_transform(df['Label'])
    
    print(f"Data dimensions (Optimized): {X.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # 2. Train Engine
    engine = PCTProEngine(d_model=64)
    engine.fit(X_train, y_train)
    
    # Get probabilities for ROC and PRC
    # XGBClassifier predict_proba is needed
    features_test = engine._synthesize_features(X_test)
    y_probs = engine.model.predict_proba(features_test)[:, 1]
    y_preds = engine.model.predict(features_test)

    # Create directory for assets if not exists
    os.makedirs('assets', exist_ok=True)

    # --- PLOT 1: Confusion Matrix ---
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('PCT-PRO: Confusion Matrix (WUSTL-EHMS)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('assets/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- PLOT 2: ROC/AUC Curve ---
    print("Generating ROC Curve...")
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'PCT-PRO ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Curve: PCT-PRO Performance')
    plt.legend(loc="lower right")
    plt.savefig('assets/roc_auc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- PLOT 3: Precision-Recall Curve ---
    print("Generating Precision-Recall Curve...")
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    ap = average_precision_score(y_test, y_probs)
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, color='b', alpha=0.8, where='post', lw=3, label=f'AP = {ap:.4f}')
    plt.fill_between(recall, precision, alpha=0.2, color='b', step='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve: PCT-PRO Reliability')
    plt.legend(loc="upper right")
    plt.savefig('assets/precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- PLOT 4: Feature Importance Heatmap (Phylogenetic Insights) ---
    print("Generating Feature Importance Heatmap...")
    importance = engine.model.feature_importances_
    # We take top 20 features for readability
    indices = np.argsort(importance)[-20:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importance[indices], align='center', color='teal')
    plt.yticks(range(len(indices)), indices) # Using indices as proxies for clade IDs
    plt.xlabel('Relative Importance')
    plt.title('Phylogenetic Feature Importance (Top Clades)')
    plt.savefig('assets/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n✅ All world-class visualizations generated in 'assets/' directory.")

if __name__ == "__main__":
    generate_research_plots()
