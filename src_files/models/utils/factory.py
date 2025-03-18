import logging
import os
from urllib import request

import torch

from ...ml_decoder.ml_decoder import add_ml_decoder_head

logger = logging.getLogger(__name__)

from ..tresnet import TResnetM, TResnetL, TResnetXL


def create_model(args, load_head=False):
    """Create a model
    """
    print(f"모델 생성 - 클래스 수: {args.num_classes}")
    
    # 먼저 기본 백본 모델 생성 (클래스 수는 최종 출력 크기에만 영향)
    model_params = {'args': args, 'num_classes': args.num_classes}
    args = model_params['args']
    args.model_name = args.model_name.lower()

    if args.model_name == 'tresnet_m':
        model = TResnetM(model_params)
    elif args.model_name == 'tresnet_l':
        model = TResnetL(model_params)
    elif args.model_name == 'tresnet_xl':
        model = TResnetXL(model_params)
    else:
        print("model: {} not found !!".format(args.model_name))
        exit(-1)
    
    # 사전 학습된 가중치 로드 - 백본만
    model_path = args.model_path
    if args.model_name == 'tresnet_l' and os.path.exists("./tresnet_l.pth"):
        model_path = "./tresnet_l.pth"
        
    if model_path:
        if not os.path.exists(model_path):
            print("다운로드 중: pretrained model...")
            try:
                request.urlretrieve(args.model_path, "./tresnet_l.pth")
                model_path = "./tresnet_l.pth"
                print('다운로드 완료')
            except Exception as e:
                print(f"다운로드 실패: {e}")
                # URL이 문제인 경우 로컬에서 모델 찾기
                if os.path.exists("./models/tresnet_l.pth"):
                    model_path = "./models/tresnet_l.pth"
                    print("로컬 경로에서 모델 찾음: ./models/tresnet_l.pth")
                else:
                    print("사전 학습된 모델을 찾을 수 없습니다. 랜덤 초기화로 진행합니다.")
                    model_path = None

        # 모델 가중치 로드
        if model_path and os.path.exists(model_path):
            print(f"사전 학습된 모델 로드 중: {model_path}")
            state = torch.load(model_path, map_location='cpu')
            
            # state_dict 추출
            if 'model' in state:
                key = 'model'
            else:
                key = 'state_dict'
                
            # 백본 가중치만 로드 (분류 헤드 제외)
            filtered_dict = {k: v for k, v in state[key].items() if
                            (k in model.state_dict() and 'head.fc' not in k)}
            
            # 가중치 로드
            missing, unexpected = model.load_state_dict(filtered_dict, strict=False)
            print(f"백본 가중치 로드 완료 - 누락된 키: {len(missing)}, 예상치 못한 키: {len(unexpected)}")
    
    # ML-Decoder 추가 (클래스 수는 여기서 설정)
    if args.use_ml_decoder:
        print(f"ML-Decoder 헤드 추가 중 - 클래스 수: {args.num_classes}")
        model = add_ml_decoder_head(model, num_classes=args.num_classes,
                                   num_of_groups=args.num_of_groups,
                                   decoder_embedding=args.decoder_embedding, 
                                   zsl=args.zsl)
        print("ML-Decoder 헤드 추가 완료")

    return model
