# ---------- Base Python ----------
FROM python:3.10-slim

# ---------- System dependencies for OCP, gmsh, OpenCascade ----------
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglu1-mesa \
    libx11-6 \
    libxext6 \
    libxcursor1 \
    libxft2 \
    libxinerama1 \
    libxrandr2 \
    libxi6 \
    libsm6 \
    libice6 \
    libxrender1 \
    libfontconfig1 \
    libfreetype6 \
    libxkbcommon0 \
    build-essential \
    gmsh \
    wget curl unzip \
    && rm -rf /var/lib/apt/lists/*

# ---------- Fix pip version (pip>24 breaks OCP installation) ----------
RUN pip install --upgrade pip
RUN pip install pip==23.2.1
RUN pip install --upgrade setuptools wheel packaging

# ---------- Install OCP (OpenCascade Python Bindings) ----------
RUN pip install --pre ocp==0.1.9

# ---------- Install your Python dependencies ----------
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# ---------- Copy source code ----------
COPY . /app

# ---------- Expose port for Gradio ----------
EXPOSE 7860

# ---------- Run app ----------
CMD ["python3", "app.py"]
