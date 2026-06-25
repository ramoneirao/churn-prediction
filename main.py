from src.data.descompactar import extrair_dados
from src.data.make_dataset import limpar_dados_brutos
from src.features.build_features import criar_features_e_preprocessar
from src.modelling.train_model import treinar_e_avaliar_modelo
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    logger.info("==========================================")
    logger.info(" Iniciando o Pipeline - Churn Prediction  ")
    logger.info("==========================================")
    
    # Etapa 1: Extrair e preparar os dados brutos
    logger.info("--- Etapa 1: Extração ---")
    extrair_dados()
    
    # Etapa 2: Limpeza dos dados
    logger.info("--- Etapa 2: Limpeza de Dados ---")
    limpar_dados_brutos(
        input_path="data/interim/BankChurners.csv",
        output_path="data/interim/cleaned_data.csv"
    )
    
    # Etapa 3: Engenharia de Features e Pré-processamento
    logger.info("--- Etapa 3: Feature Engineering ---")
    criar_features_e_preprocessar(
        input_path="data/interim/cleaned_data.csv",
        output_train="data/processed/train.csv",
        output_test="data/processed/test.csv"
    )
    
    # Etapa 4: Treinamento e Avaliação do Modelo
    logger.info("--- Etapa 4: Modelagem ---")
    treinar_e_avaliar_modelo(
        train_path="data/processed/train.csv",
        test_path="data/processed/test.csv",
        model_output_path="models/lightgbm_model.pkl"
    )
    
    logger.info("Pipeline executado com sucesso.")

if __name__ == "__main__":
    main()