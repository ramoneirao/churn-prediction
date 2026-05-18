import os
from pathlib import Path
import subprocess

def main():
    base_dir = Path(".")
    
    # Diretórios a serem criados
    dirs = [
        "data/raw",
        "data/interim",
        "data/processed",
        "notebooks",
        "src/data",
        "src/features",
        "src/models",
        "src/utils",
        "models",
    ]
    
    print("Criando diretórios...")
    for d in dirs:
        dir_path = base_dir / d
        dir_path.mkdir(parents=True, exist_ok=True)
        # Cria um arquivo .gitkeep para o git rastrear pastas vazias
        (dir_path / ".gitkeep").touch()
        print(f"  - {d}/ criado.")
        
    # Arquivos Python base
    files = [
        "src/__init__.py",
        "src/data/__init__.py",
        "src/data/make_dataset.py",
        "src/features/__init__.py",
        "src/features/build_features.py",
        "src/models/__init__.py",
        "src/models/train_model.py",
        "src/models/predict_model.py",
        "src/utils/__init__.py",
        "notebooks/01_analise_exploratoria.ipynb"
    ]
    
    print("\nCriando arquivos base...")
    for f in files:
        file_path = base_dir / f
        file_path.touch(exist_ok=True)
        print(f"  - {f} criado.")

    # Criando .gitignore
    gitignore_content = """# Data & Models
data/
!data/.gitkeep
!data/raw/.gitkeep
!data/interim/.gitkeep
!data/processed/.gitkeep
models/
!models/.gitkeep

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/
.eggs/

# Jupyter
.ipynb_checkpoints

# Virtual Environment (uv / venv)
.venv/
"""
    print("\nCriando .gitignore...")
    with open(base_dir / ".gitignore", "w") as f:
        f.write(gitignore_content)

    print("\nInicializando o gerenciamento de dependências com uv...")
    try:
        # Verifica se o uv está instalado
        subprocess.run(["uv", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Inicializa o projeto (cria pyproject.toml se não existir)
        if not os.path.exists(base_dir / "pyproject.toml"):
            # uv init configura o projeto atual
            subprocess.run(["uv", "init"], check=True)
            print("  - Projeto inicializado com 'uv init'.")
        
        # Adiciona algumas dependências padrão de Data Science
        print("  - Adicionando dependências base (pandas, scikit-learn, jupyter)...")
        subprocess.run(
            ["uv", "add", "pandas", "scikit-learn", "jupyter", "matplotlib", "seaborn"], 
            check=True
        )
        print("  - Dependências adicionadas com sucesso via uv.")
        
    except FileNotFoundError:
        print("\n[AVISO] 'uv' não encontrado no sistema. Por favor, instale o uv (https://docs.astral.sh/uv/) e rode manualmente:")
        print("  uv init")
        print("  uv add pandas scikit-learn jupyter matplotlib seaborn")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERRO] Falha ao rodar comandos do uv: {e}")

    print("\nEstrutura finalizada com sucesso! Você já pode começar o seu projeto.")

if __name__ == "__main__":
    main()
