# Descompactar o Dataset
import sys
import zipfile
from pathlib import Path

# Adiciona a pasta "src" ao sys.path 
# O linter/IDE (Pylance/Pyright) por padrão entende que "src" é a raiz dos imports
src_path = Path(__file__).resolve().parents[1]
sys.path.append(str(src_path))

from utils.logger import get_logger

logger = get_logger(__name__)

arquivo = "data/raw/archive.zip"
destino = "data/interim"

logger.info(f"Iniciando a extração do arquivo {arquivo}...")

try:
    with zipfile.ZipFile(arquivo, "r") as zip_ref:
        zip_ref.extractall(destino)
        logger.info(f"Arquivos extraídos com sucesso para {destino}!")
except FileNotFoundError:
    logger.error(f"Arquivo não encontrado: {arquivo}")
except Exception as e:
    logger.error(f"Ocorreu um erro durante a extração: {e}")

    