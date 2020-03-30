FROM nvidia/cuda:10.1-base

RUN apt-get update

# Install requirements
RUN apt-get install python3 python3-pip git libsm6 libxext6 libxrender1 -y

# Install requirements
RUN pip3 install torch torchvision numpy matplotlib opencv-python plyfile