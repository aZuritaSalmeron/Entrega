FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Puerto por defecto de Streamlit
EXPOSE 8501

# Comando para ejecutar tu interfaz y exponerla al exterior
CMD ["streamlit", "run", "TFM6_FrontEnd.py", "--server.port=8501", "--server.address=0.0.0.0"]