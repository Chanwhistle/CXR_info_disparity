#!/usr/bin/env python

import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, average_precision_score
from torch.utils.data import DataLoader
from core.utils import get_args, load_data
import warnings
from core.dataloader import VLM_Dataset
    

warnings.filterwarnings("ignore", category=UserWarning)

def collate_fn(batch):
    """
    Return batches unchanged.
    """
    return batch

def extract_text_from_dataset(dataset):
    """
    Extract text and labels.
    
    Args:
        dataset: VLM dataset.
    
    Returns:
        texts: Input texts.
        labels: Target labels.
    """
    texts = []
    labels = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        
        # Read the discharge note.
        text = sample.get("text", "") or ""
        
        # Add personal information.
        pi = sample.get("personal_information", {})
        if pi:
            pi_text = ""
            if pi.get("race"):
                pi_text += f"race: {pi['race']} "
            if pi.get("age"):
                pi_text += f"age: {pi['age']} "
            if pi_text:
                text = f"{pi_text}{text}"
        
        # Add the radiology report.
        if sample.get("rad_report"):
            text = f"{text} {sample['rad_report']}"
        
        texts.append(text if text else " ")  # Avoid empty text.
        labels.append(sample['label'])
    
    return texts, labels

def extract_features(dataset, args, vectorizer=None, fit=True):
    """
    Extract TF-IDF features.
    
    Args:
        dataset: VLM dataset.
        args: Runtime arguments.
        vectorizer: Existing vectorizer.
        fit: Fit when true.
    
    Returns:
        features: TF-IDF features.
        labels: Target labels.
        vectorizer: Fitted vectorizer.
    """
    # Extract text and labels.
    print(f"Extracting text from dataset (size: {len(dataset)})...")
    texts, labels = extract_text_from_dataset(dataset)
    
    # Create the vectorizer.
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=5000,  # Keep the top terms.
            min_df=2,  # Require two documents.
            max_df=0.8,  # Drop common terms.
            ngram_range=(1, 2),  # unigram + bigram
            sublinear_tf=True,  # log scaling
            strip_accents='unicode',
            lowercase=True
        )
    
    # Transform text.
    if fit:
        features = vectorizer.fit_transform(texts)
        print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")
    else:
        features = vectorizer.transform(texts)
    
    # Convert to a dense array.
    features = features.toarray()
    labels = np.array(labels)
    
    return features, labels, vectorizer

def compute_metrics(y_true, y_pred, y_pred_proba):
    """
    Compute evaluation metrics.
    """
    auroc = roc_auc_score(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )
    
    return {
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train(args):
    # Load data.
    train_data = load_data(args.train_data_path, args.summary_type)
    eval_data = load_data(args.dev_data_path, args.summary_type)
    test_data = load_data(args.test_data_path, args.summary_type)
    
    print(f"TOTAL TRAIN DATASET : {len(train_data)}")
    print(f"TOTAL EVAL  DATASET : {len(eval_data)}")
    print(f"TOTAL TEST  DATASET : {len(test_data)}")
    
    if args.debug:
        train_data = train_data[:20]
        eval_data = eval_data[:5]
        test_data = test_data[:5]

    # Build text-only datasets.
    print("\nCreating datasets...")
    train_dataset = VLM_Dataset(
        args=args,
        data_list=train_data,
        metadata_image_path=args.train_metadata_image_path,
        use_cxr_image=False,
        use_rad_report=args.use_rad_report if hasattr(args, 'use_rad_report') else False,
        use_discharge_note=True,
        shuffle=True,
        summarize=False
    )
    
    eval_dataset = VLM_Dataset(
        args=args,
        data_list=eval_data,
        metadata_image_path=args.dev_metadata_image_path,
        use_cxr_image=False,
        use_rad_report=args.use_rad_report if hasattr(args, 'use_rad_report') else False,
        use_discharge_note=True,
        shuffle=False,
        summarize=False
    )
    
    test_dataset = VLM_Dataset(
        args=args,
        data_list=test_data,
        metadata_image_path=args.test_metadata_image_path,
        use_cxr_image=False,
        use_rad_report=args.use_rad_report if hasattr(args, 'use_rad_report') else False,
        use_discharge_note=True,
        shuffle=False,
        summarize=False
    )
    
    # Extract TF-IDF features.
    print("\nExtracting TF-IDF features from training data...")
    X_train, y_train, vectorizer = extract_features(train_dataset, args, vectorizer=None, fit=True)
    
    print("Extracting TF-IDF features from evaluation data...")
    X_eval, y_eval, _ = extract_features(eval_dataset, args, vectorizer=vectorizer, fit=False)
    
    print("Extracting TF-IDF features from test data...")
    X_test, y_test, _ = extract_features(test_dataset, args, vectorizer=vectorizer, fit=False)
    
    print(f"Feature shape - Train: {X_train.shape}, Eval: {X_eval.shape}, Test: {X_test.shape}")
    
    # Feature scaling
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the SVM.
    print("\nTraining SVM model (CPU)...")
    # Read SVM settings.
    C_value = args.C if hasattr(args, 'C') else 1.0
    kernel_type = args.kernel if hasattr(args, 'kernel') else 'linear'
    
    model = SVC(
        C=C_value,
        kernel=kernel_type,
        probability=True,  # Required for AUROC.
        random_state=42,
        verbose=True,
        max_iter=-1
    )
    
    model.fit(X_train_scaled, y_train)
    print("Training completed!")
    
    # Evaluate training data.
    print("\nEvaluating on training data...")
    y_train_pred = model.predict(X_train_scaled)
    y_train_pred_proba = model.predict_proba(X_train_scaled)[:, 1]
    train_metrics = compute_metrics(y_train, y_train_pred, y_train_pred_proba)
    
    print(f"Train AUROC:     {train_metrics['auroc']:.4f}")
    print(f"Train AUPRC:     {train_metrics['auprc']:.4f}")
    print(f"Train Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"Train Precision: {train_metrics['precision']:.4f}")
    print(f"Train Recall:    {train_metrics['recall']:.4f}")
    print(f"Train F1:        {train_metrics['f1']:.4f}")
    
    # Evaluate validation data.
    print("\nEvaluating on validation data...")
    y_eval_pred = model.predict(X_eval_scaled)
    y_eval_pred_proba = model.predict_proba(X_eval_scaled)[:, 1]
    eval_metrics = compute_metrics(y_eval, y_eval_pred, y_eval_pred_proba)
    
    print(f"Eval AUROC:     {eval_metrics['auroc']:.4f}")
    print(f"Eval AUPRC:     {eval_metrics['auprc']:.4f}")
    print(f"Eval Accuracy:  {eval_metrics['accuracy']:.4f}")
    print(f"Eval Precision: {eval_metrics['precision']:.4f}")
    print(f"Eval Recall:    {eval_metrics['recall']:.4f}")
    print(f"Eval F1:        {eval_metrics['f1']:.4f}")
    
    # Evaluate test data.
    print("\nEvaluating on test data...")
    y_test_pred = model.predict(X_test_scaled)
    y_test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_pred_proba)
    
    print(f"Test AUROC:     {test_metrics['auroc']:.4f}")
    print(f"Test AUPRC:     {test_metrics['auprc']:.4f}")
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1:        {test_metrics['f1']:.4f}")
    
    # Save the model.
    os.makedirs(args.output_path, exist_ok=True)
    
    model_path = os.path.join(args.output_path, 'svm_model.pkl')
    vectorizer_path = os.path.join(args.output_path, 'tfidf_vectorizer.pkl')
    scaler_path = os.path.join(args.output_path, 'scaler.pkl')
    
    print("\nSaving models...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Model saved to: {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")
    print(f"Scaler saved to: {scaler_path}")
    
    # Save metrics.
    metrics_path = os.path.join(args.output_path, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Training Metrics:\n")
        f.write(f"================\n")
        for k, v in train_metrics.items():
            f.write(f"{k:12s}: {v:.4f}\n")
        f.write(f"\nValidation Metrics:\n")
        f.write(f"===================\n")
        for k, v in eval_metrics.items():
            f.write(f"{k:12s}: {v:.4f}\n")
        f.write(f"\nTest Metrics:\n")
        f.write(f"=============\n")
        for k, v in test_metrics.items():
            f.write(f"{k:12s}: {v:.4f}\n")
    
    print(f"Metrics saved to: {metrics_path}")
    
    return test_metrics

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_path, exist_ok=True)

    print("="*60)
    print("SVM Training Configuration (with VLM_Dataset)")
    print("="*60)
    print(f"C parameter          : {args.C if hasattr(args, 'C') else 1.0}")
    print(f"Kernel               : {args.kernel if hasattr(args, 'kernel') else 'linear'}")
    print(f"Summary type         : {args.summary_type}")
    print(f"Use Radiology note   : {args.use_rad_report}")
    print(f"Use Personal Info    : {args.use_pi if hasattr(args, 'use_pi') else False}")
    print(f"Output path          : {args.output_path}")
    print("="*60)
    
    test_metrics = train(args)
    
    print("\n" + "="*60)
    print(f"Final Test AUROC: {test_metrics['auroc']:.4f}")
    print(f"Final Test AUPRC: {test_metrics['auprc']:.4f}")
    print("="*60)
