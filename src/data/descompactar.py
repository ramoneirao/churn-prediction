import zipfile
from src.utils.logger import get_logger

logger = get_logger(__name__)

def extrair_dados(arquivo: str = "data/raw/archive.zip", destino: str = "data/interim"):
    """
    Função responsável por extrair o dataset zipado para a pasta interim.
    """
    logger.info(f"Iniciando a extração do arquivo {arquivo}...")

    try:
        with zipfile.ZipFile(arquivo, "r") as zip_ref:
            zip_ref.extractall(destino)
            logger.info(f"Arquivos extraídos com sucesso para {destino}!")
    except FileNotFoundError:
        logger.error(f"Arquivo não encontrado: {arquivo}")
    except Exception as e:
        logger.error(f"Ocorreu um erro durante a extração: {e}")

if __name__ == "__main__":
    extrair_dados()