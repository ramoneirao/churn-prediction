import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import TargetEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)

def criar_features_e_preprocessar(input_path: str, output_train: str, output_test: str) -> None:
    """
    Lê os dados limpos, cria novas features (razões e somas), 
    divide em treino/teste e aplica encoding categórico com Pipeline do Scikit-Learn.
    """
    logger.info(f"Lendo dados limpos de: {input_path}")
    df = pd.read_csv(input_path)
    
    logger.info("Criando engenharia de features...")
    # Criação de features de razão (adicionado um pequeno valor epsilon para evitar divisão por zero)
    df['ratio_trans_amt_dep'] = df['total_trans_amt'] / (df['dependent_count'] + 1e-5)
    df['ratio_trans_amt_ct'] = df['total_trans_amt'] / (df['total_trans_ct'] + 1e-5)
    
    # Criação de feature de soma
    df['total_spending'] = df['total_trans_amt'] + df['total_revolving_bal']
    
    logger.info("Realizando divisão de treino e teste...")
    X = df.drop(columns=['churn_flag'])
    y = df['churn_flag']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    logger.info("Configurando e aplicando o ColumnTransformer...")
    ordinal_cols = ['education_level', 'income_category', 'card_category']
    target_cols = ['marital_status']
    
    # Filtrar apenas as colunas que realmente existem no df (segurança)
    ordinal_cols = [col for col in ordinal_cols if col in X_train.columns]
    target_cols = [col for col in target_cols if col in X_train.columns]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols),
            ('tgt', TargetEncoder(), target_cols)
        ],
        remainder='passthrough'
    )
    
    # Configura o transformer para retornar DataFrames do pandas (suportado no scikit-learn >= 1.2)
    preprocessor.set_output(transform="pandas")
    
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Reintegrar as variáveis preditoras processadas com a variável alvo
    train_df = pd.concat([X_train_processed, y_train], axis=1)
    test_df = pd.concat([X_test_processed, y_test], axis=1)
    
    logger.info(f"Salvando conjunto de treino processado em: {output_train}")
    train_df.to_csv(output_train, index=False)
    
    logger.info(f"Salvando conjunto de teste processado em: {output_test}")
    test_df.to_csv(output_test, index=False)
    logger.info("Processamento e extração de features concluídos com sucesso.")
