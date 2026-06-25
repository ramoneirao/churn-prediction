from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Configurações do artefato do modelo
MODEL_PATH = "models/lightgbm_model.pkl"

app = FastAPI(
    title="API de Previsão de Churn",
    description="API em tempo real para inferência do modelo LightGBM.",
    version="1.0.0"
)

# Carrega o modelo uma única vez quando a API sobe (evita latência em cada requisição)
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    logger.info("Modelo LightGBM carregado com sucesso em memória para a API.")
else:
    logger.warning("Modelo não encontrado! Treine o modelo primeiro executando o main.py.")

# Esquema de Entrada (Input Payload) usando Pydantic
class ChurnFeatures(BaseModel):
    # Aqui definimos as features exatamente como o modelo espera.
    customer_age: int
    gender: int
    dependent_count: int
    education_level: float
    marital_status: float
    income_category: float
    card_category: float
    months_on_book: int
    total_relationship_count: int
    months_inactive_12_mon: int
    contacts_count_12_mon: int
    credit_limit: float
    total_revolving_bal: float
    avg_open_to_buy: float
    total_amt_chng_q4_q1: float
    total_trans_amt: float
    total_trans_ct: int
    total_ct_chng_q4_q1: float
    avg_utilization_ratio: float
    ratio_trans_amt_dep: float
    ratio_trans_amt_ct: float
    total_spending: float

@app.post("/predict")
def predict_churn(features: ChurnFeatures):
    """
    Recebe um JSON com os dados de um cliente e retorna se ele vai cancelar o cartão.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado no servidor.")
        
    # Converte os dados do request para um DataFrame de uma única linha
    # Usamos model_dump() (Pydantic v2) no lugar de dict()
    df_input = pd.DataFrame([features.model_dump()])
    
    try:
        # Previsão
        classe = model.predict(df_input)[0]
        prob = model.predict_proba(df_input)[0][1]
        
        return {
            "churn_predito": int(classe),
            "probabilidade_churn": float(prob),
            "status": "Risco Alto de Churn" if classe == 1 else "Cliente Retido"
        }
    except Exception as e:
        logger.error(f"Erro durante inferência: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erro interno de processamento: {str(e)}")

@app.get("/health")
def health_check():
    """Endpoint de monitoramento para orquestradores (Kubernetes, Docker) saberem se a API está viva."""
    return {"status": "online", "model_loaded": model is not None}
