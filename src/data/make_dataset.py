import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def limpar_dados_brutos(input_path: str, output_path: str) -> None:
    """
    Lê os dados brutos, remove colunas indesejadas, padroniza nomes, 
    aplica binarização e salva o dataset limpo.
    """
    logger.info(f"Lendo dados brutos de: {input_path}")
    df = pd.read_csv(input_path)
    
    # Remover a coluna CLIENTNUM caso ela exista
    if 'CLIENTNUM' in df.columns:
        df = df.drop(columns=['CLIENTNUM'])
        
    # Remover as duas últimas colunas do classificador Naive Bayes (data leakage)
    df = df.iloc[:, :-2]
    
    # Padronizar os nomes das colunas para minúsculas
    df.columns = [col.lower() for col in df.columns]
    
    # Renomear a variável alvo
    if 'attrition_flag' in df.columns:
        df = df.rename(columns={'attrition_flag': 'churn_flag'})
        
    # Binarizar a variável alvo (1 = Churn / 0 = Existente)
    if 'churn_flag' in df.columns:
        df['churn_flag'] = df['churn_flag'].map({'Attrited Customer': 1, 'Existing Customer': 0})
        
    # Binarizar a coluna Gender
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'M': 1, 'F': 0})
        
    logger.info(f"Salvando dados limpos em: {output_path}")
    df.to_csv(output_path, index=False)
    logger.info("Limpeza de dados concluída com sucesso.")
