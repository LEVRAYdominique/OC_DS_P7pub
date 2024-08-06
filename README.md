# OC_DS_P7
OpenClassrooms - Cursus Data Scientist - Projet n°7<br>
Réalisé par Dominique LEVRAY en Juillet/Août 2024

**Implémentez un modèle de scoring**

Ce dépôt contient la partie publique de ce projet et comprend :
- le script python d'une API (réalisé avec FastAPI) de prediction à publier sur un environnement heroku
- les fichiers .pkl d'un modele lightgbm entrainé pour réaliser ces prédictions
- un fichier .zip contenant des données de test
- un script python de test de l'API (réalisé avec Streamlit)
- des fichiers de gestion dont :
  - .slugignore pour contrôler quels fichiers sont publiés sur heroku
  - Procfile pour contrôler le lancement de l'API sur heroku
  - requirements.txt pour reproduire l'environnement python
