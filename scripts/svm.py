#!/usr/bin/env python

import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, average_precision_score
from torch.utils.data import DataLoader
from utils import get_args, load_data
import warnings
from dataloader import VLM_Dataset
    

warnings.filterwarnings("ignore", category=UserWarning)

def collate_fn(batch):
    """
    DataLoader를 위한 collate function
    """
    return batch

def extract_text_from_dataset(dataset):
    """
    VLM_Dataset에서 텍스트와 레이블만 추출
    
    Args:
        dataset: VLM_Dataset 인스턴스
    
    Returns:
        texts: 텍스트 리스트
        labels: 레이블 리스트
    """
    texts = []
    labels = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        
        # 텍스트 추출 (discharge note가 있으면 사용, 없으면 빈 문자열)
        text = sample.get("text", "") or ""
        
        # Personal information 추가 (race, age)
        pi = sample.get("personal_information", {})
        if pi:
            pi_text = ""
            if pi.get("race"):
                pi_text += f"race: {pi['race']} "
            if pi.get("age"):
                pi_text += f"age: {pi['age']} "
            if pi_text:
                text = f"{pi_text}{text}"
        
        # Radiology report도 사용하고 싶다면 결합
        if sample.get("rad_report"):
            text = f"{text} {sample['rad_report']}"
        
        texts.append(text if text else " ")  # 빈 텍스트 방지
        labels.append(sample['label'])
    
    return texts, labels

def extract_features(dataset, args, vectorizer=None, fit=True):
    """
    VLM_Dataset에서 TF-IDF feature 추출
    
    Args:
        dataset: VLM_Dataset 인스턴스
        args: 인자
        vectorizer: 기존 TfidfVectorizer (test 시 사용)
        fit: True면 vectorizer를 fit, False면 transform만 수행
    
    Returns:
        features: TF-IDF feature 행렬
        labels: 레이블 배열
        vectorizer: 학습된 TfidfVectorizer
    """
    # 텍스트와 레이블 추출
    print(f"Extracting text from dataset (size: {len(dataset)})...")
    texts, labels = extract_text_from_dataset(dataset)
    
    # TF-IDF Vectorizer 초기화 (처음 호출 시)
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=5000,  # 상위 5000개 단어만 사용
            min_df=2,  # 최소 2번 이상 등장한 단어만
            max_df=0.8,  # 80% 이상 문서에 등장하는 단어 제외 (불용어)
            ngram_range=(1, 2),  # unigram + bigram
            sublinear_tf=True,  # log scaling
            strip_accents='unicode',
            lowercase=True
        )
    
    # TF-IDF 변환
    if fit:
        features = vectorizer.fit_transform(texts)
        print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")
    else:
        features = vectorizer.transform(texts)
    
    # Sparse matrix를 dense array로 변환
    features = features.toarray()
    labels = np.array(labels)
    
    return features, labels, vectorizer

def compute_metrics(y_true, y_pred, y_pred_proba):
    """
    평가 지표 계산
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
    # 데이터 로드
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

    # VLM_Dataset 생성
    # SVM은 텍스트만 사용하므로 이미지는 로드하지 않음 (속도 향상)
    print("\nCreating datasets...")
    train_dataset = VLM_Dataset(
        args=args,
        data_list=train_data,
        metadata_image_path=args.train_metadata_image_path,
        use_cxr_image=False,  # SVM은 이미지 미사용
        use_rad_report=args.use_rad_report if hasattr(args, 'use_rad_report') else False,  # args에 따라 설정
        use_discharge_note=True,  # discharge note는 사용
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
    
    # TF-IDF Feature 추출
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
    
    # SVM 모델 학습
    print("\nTraining SVM model (CPU)...")
    # C 파라미터: args.C가 있으면 사용, 없으면 기본값 1.0
    C_value = args.C if hasattr(args, 'C') else 1.0
    # Kernel: args.kernel이 있으면 사용, 없으면 'linear'
    kernel_type = args.kernel if hasattr(args, 'kernel') else 'linear'
    
    model = SVC(
        C=C_value,
        kernel=kernel_type,
        probability=True,  # AUROC 계산을 위해 필요
        random_state=42,
        verbose=True,
        max_iter=-1  # 제한 없음
    )
    
    model.fit(X_train_scaled, y_train)
    print("Training completed!")
    
    # 학습 데이터 평가
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
    
    # 검증 데이터 평가
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
    
    # 테스트 데이터 평가
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
    
    # 모델 저장
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
    
    # 메트릭 저장
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