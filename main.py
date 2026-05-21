from src.data.descompactar import extrair_dados
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    logger.info("==========================================")
    logger.info(" Iniciando o Pipeline - Churn Prediction  ")
    logger.info("==========================================")
    
    # Etapa 1: Extrair e preparar os dados brutos
    extrair_dados()
    
    # Próximas etapas virão aqui no futuro...
    # processar_dados()
    # treinar_modelo()
    
    logger.info("Pipeline executado com sucesso.")

if __name__ == "__main__":
    main()