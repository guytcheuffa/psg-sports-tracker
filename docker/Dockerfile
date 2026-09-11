FROM python:3.11-slim

WORKDIR /app

# README.md est requis par pyproject.toml (project.readme) : sans lui,
# "pip install -e ." echoue au moment de generer les metadonnees du package.
COPY pyproject.toml README.md ./
COPY src/ src/
COPY scripts/ scripts/
# Theme sombre custom (couleurs du club) + mode headless : sans ce dossier,
# le dashboard containerise retombe sur le theme clair par defaut de
# Streamlit, en contradiction avec toute l'identite visuelle du projet.
COPY .streamlit/ .streamlit/

RUN pip install --no-cache-dir -e .

EXPOSE 8501

CMD ["streamlit", "run", "src/psg_tracker/app/main.py", "--server.address=0.0.0.0"]
