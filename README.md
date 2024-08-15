# OC_DS_P7pub
OpenClassrooms - Cursus Data Scientist - Projet n°7<br>
Réalisé par Dominique LEVRAY en Juillet/Août 2024

**Implémentez un modèle de scoring**

Missions :
- Mission 1 :
    - Exploratory Data Analysis (EDA) + Feature Engineering sur des données issues d'un concours kaggle (clos)
    - Modélisation supervisée d’une classification binaire (crédit accordé ou refusé)
        - Mise en place de validations croisées et d'optimisation des hyperparamètres via GridsearchCV de plusieurs modèles
        - Mise en œuvre de mlflow pour sérialiser et stocker ces expérimentations de modèles dans un registre centralisé
    - Gestion du déséquilibre des classes selon 3 méthodes : Random Under-Sampling / Random Over-Sampling / Synthetic Minority Over-sampling (SMOTE)
    - Sélection du meilleur modèle et de la meilleure méthode de gestion du déséquilibre des classes
    - Etude des features importances (SHAP) globale et locale
    - Préparation du jeu de données finale
        - Réduction du jeu de données aux données importantes
        - Entrainement et sauvegarde du modèle entrainé

- Mission 2 :
    - Etude du data drift (evidently)
    - Conception d'une API en utilisant le framework « FastAPI »
    - Conception d'une application d’utilisation de l'API (IHM) en utilisant le framework « Streamlit »
    - Mise en place d'un GitHub publique pour l'API et d'un GitHub Action permettant de tester l'api comprenant 2 tâches :
        - « Lint with flake8 » pour vérifier la conformité des scripts Python avec le standard PEP8
        - « Test with pytest » pour exécuter des tests de fonctionnement de l'API
    - Déploiement de l'API sur une plateforme de cloud (heroku)
        - Mise en place d'un pipeline heroku de déploiement automatique à chaque publication d'une nouvelle version dans la branche « main » du GitHub publique


Sources :
- Les données sources sont disponible [ici](https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring/Projet+Mise+en+prod+-+home-credit-default-risk.zip)
- Les données sont décrites [ici](https://www.kaggle.com/c/home-credit-default-risk/data)


Objectifs pédagogiques du projet :

- Concevoir un déploiement continu d'un moteur d’inférence sur une plateforme Cloud
- Définir et mettre en œuvre un pipeline d’entraînement des modèles
- Définir la stratégie d’élaboration d’un modèle d’apprentissage supervisé
- Évaluer les performances des modèles d’apprentissage supervisé
- Mettre en œuvre un logiciel de version de code
- Suivre la performance d’un modèle en production et en assurer la maintenance


**Ce dépôt contient la partie publique du projet et comprend :**
(URL du dépot GitHub publique : https://github.com/LEVRAYdominique/OC_DS_P7pub)
- Le script python d'une API (réalisé avec FastAPI) de prediction à publier sur un environnement heroku
    - levray_dominique_1_api_072024.py
- Les fichiers d'un modele lightgbm entrainé pour réaliser ces prédictions
    - dans le sous-répertoire mlflow_model
- Un fichier .zip contenant des données de test
    - test_data.zip
- Un script python de test de l'API (réalisé avec Pytest)
    - test.py
- Des fichiers de gestion :
    - .gitignore pour contrôler quels fichiers sont publiés sur GitHUB
    - .slugignore pour contrôler quels fichiers sont publiés sur heroku
    - Procfile pour contrôler le lancement de l'API sur heroku
    - requirements.txt pour reproduire l'environnement python pour heroku
    - runtime.txt pour préciser à heroku quelle version de python utiliser

**Pré-requis heroku:**
- Application Heroku de type Eco dynos
    > Quota disque = 300Mo - Quota RAM = 512 Mo - 1 vCPU
- Utilisation d'Heroku-24 stack
    > Ubuntu 24.04 + Python 3.12.4
- Génération du fichier requirements depuis un wsl 2 - ubuntu 24.04 lts
    > pip freeze > requirements.txt

**Instance heroku**
- URL de l'API : [https://oc-projet-7-c21cbfffa8fb.herokuapp.com/](https://oc-projet-7-c21cbfffa8fb.herokuapp.com/)
- Swagger FastAPP de l'API : [https://oc-projet-7-c21cbfffa8fb.herokuapp.com/docs](https://oc-projet-7-c21cbfffa8fb.herokuapp.com/docs)

**Notes concernant les tests et GitHUB**
- L'API devant être en ligne (sur heroku) pour être testée, les tests sont donc fait à postériori du déploiement