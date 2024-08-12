# OC_DS_P7pub
OpenClassrooms - Cursus Data Scientist - Projet n°7<br>
Réalisé par Dominique LEVRAY en Juillet/Août 2024

**Implémentez un modèle de scoring**
Ce dépôt contient la partie publique de ce projet et comprend :
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
- L'API doit être en ligne (sur heroku) pour être testée !
    - Les tests sont donc fait à postériori du déploiement.