# PyTorch 1.13.1 CUDA 11.6 이미지
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# 비대화형 설치 환경 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3-opencv \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# 기본 ML-Decoder 패키지 설치 (NumPy 및 기타 종속성)
RUN pip install --no-cache-dir numpy==1.24.3
RUN pip install --no-cache-dir \
    ninja \
    yacs \
    cython \
    matplotlib \
    tqdm \
    pycocotools \
    randaugment \
    pandas \
    scikit-learn

# 작업 디렉터리 설정
WORKDIR /workspace

# ML-Decoder 소스 복사
COPY . /workspace/ML_Decoder
WORKDIR /workspace/ML_Decoder

# 기본 명령
CMD ["/bin/bash"]