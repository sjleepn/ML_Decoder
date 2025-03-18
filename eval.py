import os
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

import torch
import torch.utils.data
import torchvision.transforms as transforms

from src_files.helper_functions.helper_functions import CustomCocoDetection
from src_files.models import create_model

def calculate_metrics(targets, predictions, threshold=0.5, classes=None):
    """성능 지표 계산 함수"""
    # 임계값 적용해 이진 예측으로 변환
    binary_preds = predictions > threshold
    
    # 전체 정확도 계산
    acc = accuracy_score(targets.flatten(), binary_preds.flatten())
    
    # 클래스별 precision, recall, f1, support 계산
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, binary_preds, average=None, zero_division=0
    )
    
    # 클래스별 지표를 딕셔너리에 저장
    metrics = {
        'accuracy': acc,
        'class_metrics': {}
    }
    
    for i in range(len(precision)):
        class_name = classes[i] if classes is not None else f"Class {i}"
        metrics['class_metrics'][class_name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i],
            'support': support[i]
        }
    
    # 매크로 평균 계산
    metrics['macro_avg'] = {
        'precision': np.mean(precision),
        'recall': np.mean(recall),
        'f1_score': np.mean(f1)
    }
    
    # 가중 평균 계산
    weights = support / np.sum(support)
    metrics['weighted_avg'] = {
        'precision': np.sum(precision * weights),
        'recall': np.sum(recall * weights),
        'f1_score': np.sum(f1 * weights)
    }
    
    return metrics, binary_preds

def plot_confusion_matrix(cm, classes, output_path='confusion_matrix.png'):
    """혼동 행렬 시각화"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Dog Fur Color Classification Model Evaluation')
    parser.add_argument('--model-path', default='./models/pet-colors-best.pt', type=str,
                        help='path to trained model')
    parser.add_argument('--data', default='/home/lucas/datasets/pet_color_coco_250318', type=str,
                        help='path to dataset')
    parser.add_argument('--split', default='test', type=str, choices=['val', 'test'],
                        help='dataset split to evaluate on')
    parser.add_argument('--batch-size', default=16, type=int,
                        help='batch size for evaluation')
    parser.add_argument('--num-classes', default=6, type=int,
                        help='number of classes')
    parser.add_argument('--workers', default=4, type=int,
                        help='number of workers for data loading')
    parser.add_argument('--image-size', default=448, type=int,
                        help='image size')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='confidence threshold for predictions')
    parser.add_argument('--use-ml-decoder', default=1, type=int,
                        help='use ML-Decoder')
    parser.add_argument('--num-of-groups', default=-1, type=int)
    parser.add_argument('--decoder-embedding', default=768, type=int)
    parser.add_argument('--zsl', default=0, type=int)
    parser.add_argument('--model-name', default='tresnet_l', type=str,
                        help='model architecture')
    parser.add_argument('--output-dir', default='./eval_results', type=str,
                        help='directory to save evaluation results')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Evaluating model: {args.model_path}")
    print(f"Dataset: {args.data}/{args.split}")
    
    # 모델 로드
    model = create_model(args, load_head=False).cuda()
    state = torch.load(args.model_path, map_location='cpu')
    
    if isinstance(state, dict) and 'model' in state:
        model.load_state_dict(state['model'], strict=True)
        # 클래스 정보 추출
        if 'idx_to_class' in state:
            idx_to_class = state['idx_to_class']
            classes_list = [idx_to_class[i] for i in range(args.num_classes)]
            print(f"Found classes in model: {classes_list}")
        else:
            classes_list = ["white", "black", "gray", "brown", "beige"]
    else:
        model.load_state_dict(state, strict=True)
        classes_list = ["white", "black", "gray", "brown", "beige"]
    
    model.cuda().eval()
    print("Model loaded successfully")
    
    # 데이터셋 로드
    data_path = f"{args.data}/{args.split}"
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} does not exist, falling back to val set")
        data_path = f"{args.data}/val"
        args.split = "val"
    
    instances_path = os.path.join(args.data, f'annotations/instances_{args.split}.json')
    if not os.path.exists(instances_path):
        print(f"Warning: {instances_path} does not exist, falling back to val annotations")
        instances_path = os.path.join(args.data, 'annotations/instances_val.json')
    
    # 데이터셋 및 데이터로더 설정
    dataset = CustomCocoDetection(
        data_path,
        instances_path,
        transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]),
        num_classes=args.num_classes
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=args.workers,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 예측 및 타겟 수집
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = images.cuda()
            targets = targets.max(dim=1)[0]  # 객체 크기에 따라 구분된 라벨을 합침
            all_targets.append(targets.cpu().numpy())
            
            # 모델 예측
            outputs = torch.sigmoid(model(images))
            all_predictions.append(outputs.cpu().numpy())
    
    # 배치 결과 합치기
    all_targets = np.vstack(all_targets)
    all_predictions = np.vstack(all_predictions)
    
    # 성능 지표 계산
    metrics, binary_preds = calculate_metrics(all_targets, all_predictions, 
                                            threshold=args.threshold,
                                            classes=classes_list)
    
    # 결과 출력
    print("\n===== Evaluation Results =====")
    print(f"Total samples: {len(dataset)}")
    print(f"Overall accuracy: {metrics['accuracy']:.4f}")
    print("\nClass-wise metrics:")
    for class_name, class_metrics in metrics['class_metrics'].items():
        print(f"  {class_name}:")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall:    {class_metrics['recall']:.4f}")
        print(f"    F1 Score:  {class_metrics['f1_score']:.4f}")
        print(f"    Support:   {class_metrics['support']}")
    
    print("\nMacro average:")
    print(f"  Precision: {metrics['macro_avg']['precision']:.4f}")
    print(f"  Recall:    {metrics['macro_avg']['recall']:.4f}")
    print(f"  F1 Score:  {metrics['macro_avg']['f1_score']:.4f}")
    
    print("\nWeighted average:")
    print(f"  Precision: {metrics['weighted_avg']['precision']:.4f}")
    print(f"  Recall:    {metrics['weighted_avg']['recall']:.4f}")
    print(f"  F1 Score:  {metrics['weighted_avg']['f1_score']:.4f}")
    
    # 결과 파일로 저장
    result_file = os.path.join(args.output_dir, 'evaluation_results.txt')
    with open(result_file, 'w') as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Dataset: {args.data}/{args.split}\n")
        f.write(f"Total samples: {len(dataset)}\n")
        f.write(f"Overall accuracy: {metrics['accuracy']:.4f}\n\n")
        
        f.write("Class-wise metrics:\n")
        for class_name, class_metrics in metrics['class_metrics'].items():
            f.write(f"  {class_name}:\n")
            f.write(f"    Precision: {class_metrics['precision']:.4f}\n")
            f.write(f"    Recall:    {class_metrics['recall']:.4f}\n")
            f.write(f"    F1 Score:  {class_metrics['f1_score']:.4f}\n")
            f.write(f"    Support:   {class_metrics['support']}\n")
        
        f.write("\nMacro average:\n")
        f.write(f"  Precision: {metrics['macro_avg']['precision']:.4f}\n")
        f.write(f"  Recall:    {metrics['macro_avg']['recall']:.4f}\n")
        f.write(f"  F1 Score:  {metrics['macro_avg']['f1_score']:.4f}\n")
        
        f.write("\nWeighted average:\n")
        f.write(f"  Precision: {metrics['weighted_avg']['precision']:.4f}\n")
        f.write(f"  Recall:    {metrics['weighted_avg']['recall']:.4f}\n")
        f.write(f"  F1 Score:  {metrics['weighted_avg']['f1_score']:.4f}\n")
    
    print(f"Evaluation results saved to {result_file}")
    
    # 혼동 행렬 계산 및 시각화
    cm = confusion_matrix(all_targets.argmax(axis=1), binary_preds.argmax(axis=1))
    plot_confusion_matrix(cm, classes_list, 
                         output_path=os.path.join(args.output_dir, 'confusion_matrix.png'))

if __name__ == '__main__':
    main()