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
  - requirements.txt pour reproduire l'environnement python pour heroku


**Quelques valeurs de clients remarquables :**

- "Matrice de confusion": "TN=54928 - FN=4057 - FP=1720 - TP=798",
- "Quelques TN": "451879, 400678, 405872, 438462, 128526, 417574, 156652, 254016, 385475, 328019",
- "Quelques FN": "268908, 228368, 399106, 112131, 265025, 282128, 256012, 117908, 343178, 216763",
- "Quelques FP": "213490, 132606, 425613, 414718, 299349, 321130, 215611, 186593, 128286, 422979",
- "Quelques TP": "166700, 287807, 224163, 142769, 183960, 311408, 225829, 336600, 180035, 126228"


**Notes concernant les pré-requis heroku:**
    Dû a un bugg avec pip freeze et la version 20.1, le fichier requirements.txt est généré avec la commande :
        pip list --format=freeze > requirements.txt

    L'API est conçu pour fonctionner sur heroku qui est une plateforme linux
        => suppression à la main de requirements.txt des packages inutiles suivant :
            pywin32 

    Ajout d'un fichier runtime.txt pour spécifier la version de Pythony