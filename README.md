# Churn Prediction Project


## Estrutura Proposta
```
churn-prediction/
├── data/ 
│   ├── raw/ 
│   ├── interim/
│   └── processed/ 
│
├── notebooks/       
│   └── 01_analise_exploratoria.ipynb
│
├── src/             
│   ├── __init__.py
│   ├── data/        
│   │   └── make_dataset.py  
│   ├── features/      
│   │   └── build_features.py
│   ├── modelling/ 
│   │   ├── train_model.py
│   │   └── predict_model.py
│   └── utils/     
│
├── models/        
│
├── pyproject.toml 
│
├── .gitignore     
└── README.md 
```