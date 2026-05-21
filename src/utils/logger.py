import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Cria e configura um logger padrão para o projeto.
    Dessa forma você pode chamar get_logger(__name__) em qualquer arquivo
    e manter a mesma formatação em toda a aplicação.
    """
    logger = logging.getLogger(name)
    
    # Previne adicionar múltiplos handlers se o logger já for instanciado mais de uma vez
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger
    