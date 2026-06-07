# Usamos una imagen oficial de Python ligera
FROM python:3.11-slim

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos primero los requerimientos para aprovechar la caché de Docker
COPY requirements.txt .

# Instalamos las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto de tu código (TFM_AZ.py, Motor_RAG.py y la carpeta data)
COPY . .

# Exponemos el puerto de tu API Flask
EXPOSE 8000

# Comando para levantar el servidor
CMD ["python", "TFM_AZ.py"]