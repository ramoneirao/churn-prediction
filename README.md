<div align="center">

# Bank Churn Prediction

![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-F3F3F3?logo=lightgbm&logoColor=black)
![uv](https://img.shields.io/badge/uv-Package_Manager-blueviolet)

</div>

Este repositório contém um pipeline de Machine Learning de ponta a ponta para a previsão de churn (cancelamento) de clientes de cartão de crédito. O projeto está estruturado de forma modular, seguindo as melhores práticas da indústria para projetos de ciência de dados, e inclui o processamento de dados, engenharia de features, treinamento do modelo com rastreamento de experimentos (MLFlow/DagsHub) e a implantação do modelo através de uma API RESTful (FastAPI).

## Índice

1. [Visão Geral do Projeto](#visão-geral-do-projeto)
2. [Estrutura do Projeto](#estrutura-do-projeto)
3. [Descrição dos Componentes](#descrição-dos-componentes)
4. [Notebooks](#notebooks)
5. [Instalação e Configuração](#instalação-e-configuração)
6. [Como Usar](#como-usar)
7. [Testes e API](#testes-e-api)

## Visão Geral do Projeto

O objetivo principal deste projeto é identificar clientes com alto risco de cancelar seus serviços de cartão de crédito. Ao prever o churn, instituições financeiras podem oferecer incentivos de retenção de maneira proativa. O pipeline implementa:
- Extração e limpeza automatizada de dados.
- Engenharia de features, incluindo codificação de variáveis categóricas e criação de variáveis específicas do domínio.
- Treinamento de modelo utilizando LightGBM otimizado para classes desbalanceadas.
- Rastreamento de experimentos integrado com MLFlow e DagsHub.
- Uma API de inferência em tempo real construída com FastAPI.

> **Conjunto de Dados:** Os dados utilizados para o desenvolvimento deste modelo são públicos e estão disponíveis no Kaggle: [Credit Card Customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers).

## Estrutura do Projeto

```text
.
├── data
│   ├── interim/               # Dados intermediários que foram transformados
│   ├── processed/             # Dados finais prontos para a modelagem (treino/teste)
│   └── raw/                   # Dados brutos originais e imutáveis (archive.zip)
├── models/                    # Modelos treinados e serializados (.pkl)
├── notebooks/                 # Notebooks Jupyter para análise exploratória de dados (EDA)
├── src/                       # Código-fonte do projeto
│   ├── data/                  # Scripts para baixar, extrair e limpar os dados
│   ├── features/              # Scripts para criar e codificar features
│   ├── modelling/             # Scripts para treinar o modelo e expor a API
│   └── utils/                 # Scripts utilitários (ex: logs customizados)
├── main.py                    # Ponto de entrada que orquestra todo o pipeline
├── pyproject.toml             # Dependências e configurações do projeto
├── uv.lock                    # Arquivo de bloqueio (lock) de dependências do uv
└── README.md                  # Documentação do projeto
```

## Descrição dos Componentes

### Processamento de Dados (`src/data`)
- **`descompactar.py`**: Lida com a extração dos dados brutos do arquivo compactado para o diretório "interim" (intermediário) para processamento futuro.
- **`make_dataset.py`**: Responsável pela limpeza inicial dos dados, que inclui a remoção de identificadores irrelevantes (como `CLIENTNUM`), descarte de colunas que causam vazamento de dados (data leakage), padronização dos nomes das colunas e binarização de variáveis alvo.

### Engenharia de Features (`src/features`)
- **`build_features.py`**: Executa a lógica de engenharia de features. Cria novas features derivadas (ex: proporções de valores de transações e métricas totais de gastos), divide o conjunto de dados em treino e teste, e aplica o `ColumnTransformer` para a codificação de variáveis categóricas (Ordinal Encoding e Target Encoding).

### Modelagem e Inferência (`src/modelling`)
- **`train_model.py`**: Orquestra o treinamento do modelo utilizando `LightGBMClassifier`. Conecta-se ao MLFlow/DagsHub para registrar hiperparâmetros, métricas (ROC-AUC e PR-AUC) e o artefato do modelo resultante. Por fim, serializa o modelo treinado localmente no diretório `models/`.
- **`predict_model.py`**: Um servidor web em FastAPI que carrega o artefato do modelo treinado e expõe um endpoint `/predict`. Aceita um payload JSON correspondente às features do cliente e retorna a probabilidade de churn e o status de risco.

### Utilitários (`src/utils`)
- **`logger.py`**: Configura um logger Python padronizado, utilizado em todos os módulos para garantir o registro consistente de logs na saída padrão do terminal.

## Notebooks

Os notebooks Jupyter foram utilizados para a fase de pesquisa, exploração de dados e experimentação de modelos antes da consolidação do código nos scripts de produção.

- **`01_analise_exploratoria.ipynb`**: Focado na Análise Exploratória de Dados (EDA). Os principais passos realizados incluem:
  - Entendimento do problema de negócio e definição de KPIs estratégicos.
  - Carregamento, inspeção inicial e análise descritiva dos dados.
  - Verificação de qualidade dos dados (valores ausentes, duplicados e inconsistências).
  - Extração de *insights* e identificação de padrões comportamentais de clientes propensos ao cancelamento (*churners*).

- **`02_modelagem.ipynb`**: Focado na Preparação dos Dados, Modelagem e Avaliação. Os principais passos realizados incluem:
  - Limpeza e pré-processamento dos dados (ex: remoção de identificadores e variáveis com vazamento de dados).
  - Engenharia de *features* e codificação de variáveis categóricas (*Target Encoding*, *Ordinal Encoding*).
  - Divisão do conjunto de dados em bases de treino e teste.
  - Treinamento, experimentação e validação de algoritmos de *Machine Learning* (LightGBM).
  - Avaliação do desempenho do modelo utilizando métricas apropriadas para dados desbalanceados (ROC-AUC e PR-AUC).

## Instalação e Configuração

Este projeto utiliza o `uv` como seu gerenciador de pacotes e dependências. Certifique-se de ter o `uv` instalado em seu ambiente antes de prosseguir.

1. Clone o repositório e navegue até o diretório do projeto:
```bash
git clone https://github.com/ramoneirao/churn-prediction
cd churn-prediction
```

2. Sincronize o ambiente e instale as dependências:
```bash
uv sync
```

## Como Usar

> **⚠️ Atenção sobre o DagsHub:** Se você for executar o pipeline de treinamento, lembre-se que o script `src/modelling/train_model.py` está configurado para enviar os registros do MLFlow para o meu repositório no x   DagsHub (`ramoneirao`).
> Antes de executar, abra o arquivo `src/modelling/train_model.py` e altere o parâmetro `repo_owner` na inicialização do DagsHub para o **seu próprio nome de usuário** (ou remova a linha se quiser rodar apenas o MLFlow localmente):
> ```python
> dagshub.init(repo_owner='SEU_USUARIO_AQUI', repo_name='churn-prediction', mlflow=True)
> ```

Para executar todo o pipeline de Machine Learning de ponta a ponta, você pode rodar o script orquestrador principal. Isso irá disparar sequencialmente a extração de dados, limpeza, engenharia de features e o treinamento do modelo.

```bash
uv run main.py
```

Acompanhe os logs na saída padrão (terminal) para monitorar o progresso do pipeline. Após a conclusão bem-sucedida, os conjuntos de dados limpos e processados estarão disponíveis em `data/processed/`, e o artefato do modelo treinado será salvo em `models/lightgbm_model.pkl`.

## Testes e API

Uma vez que o modelo tenha sido treinado e salvo, você pode iniciar o servidor FastAPI para lidar com previsões em tempo real.

### 1. Iniciar o Servidor da API

Utilize o `uvicorn` por meio do `uv run` para iniciar o servidor web:

```bash
uv run uvicorn src.modelling.predict_model:app --reload
```

A API será inicializada e ficará disponível localmente em `http://127.0.0.1:8000`.

### 2. Testando via Swagger UI

O FastAPI fornece automaticamente uma interface de documentação interativa da API, alimentada pelo Swagger UI. Este é o método mais direto para testar as previsões do modelo.

1. Abra o seu navegador e acesse: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
2. Expanda a seção do endpoint `POST /predict`.
3. Clique no botão **"Try it out"**.
4. Modifique o payload JSON fornecido com as features do cliente que você deseja testar.
5. Clique em **"Execute"**.
6. A resposta do servidor aparecerá logo abaixo, detalhando a `churn_predito` (classe prevista), a `probabilidade_churn` (probabilidade de churn) e o `status` (classificação de risco).

### 3. Testando via cURL

Alternativamente, você pode testar a API diretamente do seu terminal utilizando uma requisição `curl`:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "customer_age": 45,
  "gender": 1,
  "dependent_count": 3,
  "education_level": 2.0,
  "marital_status": 1.0,
  "income_category": 3.0,
  "card_category": 1.0,
  "months_on_book": 36,
  "total_relationship_count": 5,
  "months_inactive_12_mon": 1,
  "contacts_count_12_mon": 2,
  "credit_limit": 12000.0,
  "total_revolving_bal": 1500.0,
  "avg_open_to_buy": 10500.0,
  "total_amt_chng_q4_q1": 0.8,
  "total_trans_amt": 4500.0,
  "total_trans_ct": 65,
  "total_ct_chng_q4_q1": 0.7,
  "avg_utilization_ratio": 0.12,
  "ratio_trans_amt_dep": 1000.0,
  "ratio_trans_amt_ct": 50.0,
  "total_spending": 6000.0
}'
```

### 4. Deploy Alternativo via MLFlow

Além da API personalizada construída com FastAPI, o projeto registra o modelo treinado diretamente no **MLFlow** (integrado ao DagsHub). Se preferir utilizar o servidor de inferência nativo do MLFlow, você pode realizar o *deploy* local executando:

```bash
mlflow models serve -m "runs:/<RUN_ID>/lightgbm-model" -p 5001 --env-manager local
```

> **Nota:** Substitua `<RUN_ID>` pelo ID da run gerada no terminal ou visualizada no dashboard do MLFlow/DagsHub após o treinamento.

O modelo ficará disponível localmente na porta 5001 e você poderá enviar requisições `POST` para a rota `/invocations` enviando os dados em formato JSON padrão de DataFrames (ex: *split* orient).


## Contribuição

Contribuições são muito bem-vindas! Sinta-se à vontade para abrir **Issues** para relatar bugs ou sugerir melhorias, e **Pull Requests** para enviar código.

1.  Faça um Fork do projeto.
2.  Crie uma Branch para sua feature (`git checkout -b feature/MinhaFeature`).
3.  Commit suas mudanças (`git commit -m 'feat:  add minha feature'`).
4.  Push para a Branch (`git push origin feature/MinhaFeature`).
5.  Abra um Pull Request.


---

<p align="center">
  Feito por <a href="https://github.com/ramoneirao">Ramon</a>
</p>
