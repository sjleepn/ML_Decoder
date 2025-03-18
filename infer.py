import os
import argparse
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed

from src_files.helper_functions.bn_fusion import fuse_bn_recursively
from src_files.models import create_model
import matplotlib

from src_files.models.tresnet.tresnet import InplacABN_to_ABN

# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# 인자 기본값 수정
parser = argparse.ArgumentParser(description='강아지 털 색상 분류 추론')
parser.add_argument('--num-classes', default=5, type=int)
parser.add_argument('--model-path', type=str, default='./models/dog-colors-best.pt')
parser.add_argument('--pic-path', type=str, default='/home/lucas/datasets/dog_color_coco_250317/test/88e12121-7ad0-4dae-83f1-a8e9a24014a1.jpeg')
parser.add_argument('--model-name', type=str, default='tresnet_l')
parser.add_argument('--image-size', type=int, default=448)
# parser.add_argument('--dataset-type', type=str, default='MS-COCO')
parser.add_argument('--th', type=float, default=0.75)
parser.add_argument('--top-k', type=float, default=20)
# ML-Decoder
parser.add_argument('--use-ml-decoder', default=1, type=int)
parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
parser.add_argument('--decoder-embedding', default=768, type=int)
parser.add_argument('--zsl', default=0, type=int)

# 메인 함수의 시작 부분 문구도 변경
def main():
    print('강아지 털 색상 분류 추론 코드')

    # parsing args
    args = parser.parse_args()

    # Setup model
    print('creating model {}...'.format(args.model_name))
    
    # load_head=False로 수정
    model = create_model(args, load_head=False).cuda()
    
    state = torch.load(args.model_path, map_location='cpu')

    # 모델 로드 부분 수정
    if isinstance(state, dict) and 'model' in state:
        # 체크포인트가 {'model': state_dict, ...} 형식인 경우
        model.load_state_dict(state['model'], strict=True)
    else:
        # 체크포인트가 직접 state_dict인 경우 (현재 우리 모델)
        model.load_state_dict(state, strict=True)

    ########### eliminate BN for faster inference ###########
    model = model.cpu()
    model = InplacABN_to_ABN(model)
    model = fuse_bn_recursively(model)
    model = model.cuda().half().eval()
    #######################################################
    print('done')


    # 클래스 목록 생성 부분 수정
    if isinstance(state, dict) and 'idx_to_class' in state:
        # 체크포인트에 클래스 정보가 있는 경우
        classes_list = np.array(list(state['idx_to_class'].values()))
    else:
        # 클래스 정보가 없는 경우, 수동으로 정의
        if args.num_classes == 80:  # COCO
            # COCO 클래스 목록 - 파일에서 불러오거나 하드코딩
            COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 
    'bus', 'train', 'truck', 'boat', 'traffic light', 
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
            classes_list = np.array(COCO_CLASSES)
        elif args.num_classes == 5:  # 강아지 털 색상
            classes_list = np.array(["white", "black", "gray", "brown", "beige"])
        else:
            # 일반적인 경우: 0부터 num_classes-1까지 번호 부여
            classes_list = np.array([f"class_{i}" for i in range(args.num_classes)])

    print('done\n')

    # doing inference
    print('loading image and doing inference...')
    im = Image.open(args.pic_path)
    im_resize = im.resize((args.image_size, args.image_size))
    np_img = np.array(im_resize, dtype=np.uint8)
    tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
    tensor_batch = torch.unsqueeze(tensor_img, 0).cuda().half() # float16 inference
    output = torch.squeeze(torch.sigmoid(model(tensor_batch)))
    np_output = output.cpu().detach().numpy()


    ## Top-k predictions
    # detected_classes = classes_list[np_output > args.th]
    idx_sort = np.argsort(-np_output)
    detected_classes = np.array(classes_list)[idx_sort][: args.top_k]
    scores = np_output[idx_sort][: args.top_k]
    idx_th = scores > args.th
    detected_classes = detected_classes[idx_th]
    print('done\n')

    # displaying image
    print('showing image on screen...')
    fig = plt.figure()
    plt.imshow(im)
    plt.axis('off')
    plt.axis('tight')
    plt.title("Detected dog fur colors: {}".format(detected_classes))
    # 화면 표시 대신 파일로 저장
    plt.savefig('output_prediction.jpg')
    print(f"Prediction result saved to output_prediction.jpg")

    # 결과 터미널에 출력 - 색상 특화 메시지
    print("\nDetected fur colors:")
    for i, (cls, score) in enumerate(zip(detected_classes, scores[:len(detected_classes)])):
        print(f"  {i+1}. {cls}: {score*100:.2f}%")
    print('done\n')


if __name__ == '__main__':
    main()
