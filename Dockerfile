# =========================
# 1. Base image
# =========================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# =========================
# 2. System dependencies
# =========================
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev python3-distutils \
    git wget curl build-essential \
    libgl1 libgl1-mesa-dev libglu1-mesa libxrender1 libxcursor1 \
    libxext6 libxft2 libxinerama1 libxrandr2 libxi6 libsm6 libice6 \
    libfontconfig1 libfreetype6 libxkbcommon0 \
    gmsh \
    && rm -rf /var/lib/apt/lists/*

# =========================
# 3. Upgrade pip
# =========================
RUN pip3 install --upgrade pip

# =========================
# 4. Install OpenCascade bindings (OCP)
# =========================
# ocp-python 7.7 is fully compatible with Ubuntu22.04 + Python3.10 on Render
RUN pip3 install --pre ocp

# =========================
# 5. Install project Python libs
# =========================
COPY requirements.txt /workspace/requirements.txt
RUN pip3 install -r /workspace/requirements.txt

# =========================
# 6. Copy project code
# =========================
COPY . /workspace
WORKDIR /workspace

# =========================
# 7. Expose Gradio port
# =========================
EXPOSE 7860

# =========================
# 8. Run app
# =========================
CMD ["python3", "app.py"]
