FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e .

EXPOSE 8501

CMD ["streamlit", "run", "src/psg_tracker/app/main.py", "--server.address=0.0.0.0"]
