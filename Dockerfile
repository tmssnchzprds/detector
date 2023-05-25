# Imagen base de Nvidia para Jetson Nano
FROM nvcr.io/nvidia/l4t-base:r35.1.0
ENV TZ=Europe \
    DEBIAN_FRONTEND=noninteractive
# Instalar dependencias para OpenCV y YOLO
RUN echo 8 | apt-get update && apt-get upgrade -y && apt-get install -y \
    python3-pip

# Instalación de paquetes de Python
RUN pip3 install --upgrade pip
RUN pip3 install numpy opencv-contrib-python

# Copiar archivos de código y modelos al contenedor
COPY person.py /app/
COPY person-face.py /app/
COPY model/ /app/model/

# Configurar el directorio de trabajo
WORKDIR /app