import os
import argparse

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src_files.helper_functions.helper_functions import mAP, CustomCocoDetection, CutoutPIL, ModelEma, \
    add_weight_decay  # CocoDetection 대신 CustomCocoDetection 임포트
from src_files.models import create_model
from src_files.loss_functions.losses import AsymmetricLoss
from torch.cuda.amp import GradScaler, autocast

parser = argparse.ArgumentParser(description='강아지 털 색상 분류 학습')
# 기본값 수정
parser.add_argument('--data', type=str, default='/home/lucas/datasets/pet_color_coco_250318')
parser.add_argument('--num-classes', default=6, type=int)
parser.add_argument('--epochs', default=10, type=int)  # 에폭 추가
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--model-name', default='tresnet_xl')
parser.add_argument('--model-path', default='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ML_Decoder/tresnet_xl_COCO_640_91_4.pth', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--image-size', default=640, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size')

# ML-Decoder 옵션 유지
parser.add_argument('--use-ml-decoder', default=1, type=int)
parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
parser.add_argument('--decoder-embedding', default=768, type=int)
parser.add_argument('--zsl', default=0, type=int)

def main():
    args = parser.parse_args()
    
    # 강아지 색상 클래스 정의
    idx_to_class = {
        0: "white",
        1: "black", 
        2: "gray",
        3: "taupe",
        4: "brown",
        5: "beige"
    }
    print(f"Dataset classes: {idx_to_class}")  # 영문으로 변경

    # 모델 생성
    print('모델 생성: {}...'.format(args.model_name))
    model = create_model(args).cuda()
    print('모델 생성 완료')

    # 데이터 경로 설정
    instances_path_val = os.path.join(args.data, 'annotations/instances_val.json')
    instances_path_train = os.path.join(args.data, 'annotations/instances_train.json')
    data_path_val = f'{args.data}/val'
    data_path_train = f'{args.data}/train'
    
    # 커스텀 데이터셋 로더 사용 (num_classes=5로 설정)
    val_dataset = CustomCocoDetection(data_path_val,
                                instances_path_val,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                ]),
                                num_classes=args.num_classes)  # 클래스 수 지정
                                
    train_dataset = CustomCocoDetection(data_path_train,
                                  instances_path_train,
                                  transforms.Compose([
                                      transforms.Resize((args.image_size, args.image_size)),
                                      CutoutPIL(cutout_factor=0.5),
                                      transforms.ToTensor(),
                                  ]),
                                  num_classes=args.num_classes)  # 클래스 수 지정
                                  
    print("Val dataset size: ", len(val_dataset))
    print("Train dataset size: ", len(train_dataset))

    # 데이터 로더 설정
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # 학습 실행
    train_multi_label_coco(model, train_loader, val_loader, args.lr, args.epochs, idx_to_class)

def train_multi_label_coco(model, train_loader, val_loader, lr, epochs, idx_to_class):
    # EMA 모델 설정
    ema = ModelEma(model, 0.9997)
    
    # 옵티마이저 설정
    weight_decay = 1e-4
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                      pct_start=0.2)

    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    
    # 학습 루프
    for epoch in range(epochs):
        model.train()
        for i, (inputData, target) in enumerate(train_loader):
            # 디버깅 출력 영문으로 변경
            if i == 0 and epoch == 0:
                print(f"Input data shape: {inputData.shape}")
                print(f"Target shape: {target.shape}")
                
            # 데이터와 타겟을 GPU로
            inputData = inputData.cuda()
            target = target.cuda()
            target = target.max(dim=1)[0]  # 객체 크기에 따라 구분된 라벨을 합침
            
            # 첫 배치 디버깅 출력
            if i == 0 and epoch == 0:
                print(f"Target shape after transformation: {target.shape}")
            
            # 혼합 정밀도로 순전파
            with autocast():
                output = model(inputData).float()
                
                # 첫 배치 디버깅 출력
                if i == 0 and epoch == 0:
                    print(f"Model output shape: {output.shape}")
                    # 출력과 타겟 차원 확인
                    if output.shape[1] != target.shape[1]:
                        print(f"[Warning] Output dimension mismatch: {output.shape[1]} != {target.shape[1]}")
            
            # loss 계산
            loss = criterion(output, target)
            model.zero_grad()
            
            # 역전파 및 가중치 업데이트
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            
            if i % 20 == 0:
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.4f}'
                      .format(epoch, epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0], loss.item()))

        # 에폭 종료 시 모델 저장
        try:
            os.makedirs('models', exist_ok=True)
            
            # 일반 체크포인트
            torch.save(model.state_dict(), 
                    os.path.join('models/', f'dog-colors-epoch{epoch + 1}.ckpt'))
            
            # 테스트 및 최고 성능 모델 저장
            model.eval()
            mAP_score = validate_multi(val_loader, model, ema)
            model.train()
            
            if mAP_score > highest_mAP:
                highest_mAP = mAP_score
                # 클래스 정보와 함께 저장
                save_dict = {
                    'model': model.state_dict(),
                    'idx_to_class': idx_to_class,
                    'mAP': highest_mAP,
                    'epoch': epoch
                }
                torch.save(save_dict, os.path.join('models/', 'pet-colors-best.pt'))
                print(f'Best model saved! mAP: {highest_mAP:.2f}')
        except Exception as e:
            print(f"모델 저장 중 오류 발생: {e}")
            
        print('Current mAP = {:.2f}, Best mAP = {:.2f}\n'.format(mAP_score, highest_mAP))  # 영문으로 변경

def validate_multi(val_loader, model, ema_model):
    print("Starting validation...")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    
    for i, (input, target) in enumerate(val_loader):
        # GPU로 이동
        input = input.cuda()
        target = target.max(dim=1)[0]  # 객체 크기에 따라 구분된 라벨을 합침
        
        # 예측 수행
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input)).cpu()
                output_ema = Sig(ema_model.module(input)).cpu()
                
        # 결과 수집
        preds_regular.append(output_regular)
        preds_ema.append(output_ema)
        targets.append(target.cpu())

    # 텐서 합치기
    targets = torch.cat(targets).numpy()
    preds_regular = torch.cat(preds_regular).numpy()
    preds_ema = torch.cat(preds_ema).numpy()

    # 성능 계산
    mAP_score_regular = mAP(targets, preds_regular)
    mAP_score_ema = mAP(targets, preds_ema)
    print("Regular model mAP: {:.2f}, EMA model mAP: {:.2f}".format(mAP_score_regular, mAP_score_ema))  # 영문으로 변경
    
    return max(mAP_score_regular, mAP_score_ema)

if __name__ == '__main__':
    main()
