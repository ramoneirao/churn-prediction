import joblib
import pandas as pd
import mlflow
import mlflow.lightgbm
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from src.utils.logger import get_logger

logger = get_logger(__name__)

def treinar_e_avaliar_modelo(train_path: str, test_path: str, model_output_path: str) -> None:
    """
    Treina um modelo LightGBM para prever churn lidando com desbalanceamento, 
    registra as métricas/artefatos no MLFlow (DagsHub) e salva o modelo no disco.
    """
    logger.info(f"Carregando dados de treino e teste...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=['churn_flag'])
    y_train = train_df['churn_flag']
    
    X_test = test_df.drop(columns=['churn_flag'])
    y_test = test_df['churn_flag']
    
    # Configuração do MLFlow apontando para o DagsHub usando a biblioteca oficial
    logger.info("Inicializando conexão com DagsHub...")
    import dagshub
    dagshub.init(repo_owner='ramoneirao', repo_name='churn-prediction', mlflow=True)
    
    mlflow.set_experiment("Bank_Churn_Prediction")
    logger.info("Iniciando a run no MLFlow...")
    with mlflow.start_run():
        
        # Hiperparâmetros base focados no problema
        params = {
            'class_weight': 'balanced',
            'random_state': 42,
            'n_estimators': 100,
            'learning_rate': 0.1
        }
        
        mlflow.log_params(params)
        
        logger.info("Treinando modelo LightGBM...")
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        
        logger.info("Calculando métricas no conjunto de teste...")
        # Coletar a probabilidade de ser classe positiva (Churn)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        
        logger.info(f"Desempenho -> ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
        
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)
        
        logger.info("Registrando artefato do modelo via MLFlow...")
        mlflow.lightgbm.log_model(model, "lightgbm-model")
        
        logger.info(f"Salvando o modelo localmente em: {model_output_path}")
        joblib.dump(model, model_output_path)
        
        logger.info("Treinamento e logging finalizados com sucesso!")
