# 8INF892 – Projet final

Ce projet explore l'application de différentes architectures de Graph Neural Network (GNN) afin de prédire des propriétés moléculaires.

## Auteur
- Maxime Simard (SIMM26050001)

## Prérequis
Pour exécuter ce projet, les prérequis suivants sont nécessaires:
- Python 3.8 ou supérieur
- PyTorch 1.8 ou supérieur
- PyTorch Geometric
- CUDA (optionnel pour l'accélération GPU)

## Installation
Suivez ces étapes pour installer et configurer l'environnement du projet:

1. Clonez le dépôt du projet
2. Installez les dépendances du projet:
   ```bash
    pip install -r requirements.txt
    ```
3. Si vous utilisez un environnement CUDA pour PyTorch, assurez-vous que la version CUDA est correctement installée et configurée sur votre système.

## Utilisation
Pour démarrer et utiliser le projet, suivez les instructions ci-dessous:

1. Assurez-vous que toutes les dépendances sont installées comme indiqué dans la section Installation.

2. Exécutez le fichier `__main__.py` pour démarrer le processus d'apprentissage ou d'évaluation. Vous pouvez passer des paramètres spécifiques en ligne de commande si nécessaire. Voici un exemple de commande pour exécuter le script:
   ```bash
   python __main__.py --mode train --model gcn
   ```
    Les paramètres suivants sont disponibles:
    - `--mode`: spécifie le mode d'exécution (train ou tune)
      - `train`: entraîne le modèle avec les hyperparamètres par défaut
      - `tune`: effectue une recherche d'hyperparamètres pour le modèle
    - `--model`: spécifie le modèle de GNN à utiliser (gcn, gat, gsage)

3. Pour modifier la recherche d'hyperparamètres lors du tuning ou pour spécifier ceux utiliser lors de l'entraînement, modifier directement les valeurs dans le fichier `__main__.py`.