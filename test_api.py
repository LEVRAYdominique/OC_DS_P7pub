'''
========================================================================
Projet n°7 - Implémentez un modèle de scoring
Script Python réalisé par Dominique LEVRAY en Juillet/Août 2024
========================================================================
Ce fichier contient un test de l'API déployée sur HEROKU
Pour exécuter ce fichier : pytest
'''
# pylint: disable=line-too-long
# pylint: disable=invalid-name
# pylint: disable=broad-exception-raised
# pylint: disable=trailing-whitespace
# pylint: disable=unused-import

# -------------------------------------------------------------------------------------------------------------------
# Importation des modules
# -------------------------------------------------------------------------------------------------------------------

import json
import requests
import pytest       # noqa: F401

# -------------------------------------------------------------------------------------------------------------------
# Définition des constantes
# -------------------------------------------------------------------------------------------------------------------

API_BASE_URL  = "https://oc-projet-7-c21cbfffa8fb.herokuapp.com"    # URL de base de l'API sur heroku
API_MATRICE   = "/matrice_confusion"
API_GET       = "/get_client/"
API_POST      = "/post_client/{Client_new_credit}"

HTTPS_TIMEOUT = 30                                                  # Timeout pour les requettes https (30 secondes)

GOOD_ID_CLIENT = 207108
BAD_ID_CLIENT  = 207109
BAD_ID_FORMAT  = 45

GOOD_DICT_CLIENT = {"SK_ID_CURR": 207108,
                    "FLAG_OWN_REALTY": 1,
                    "FLAG_OWN_CAR": 0,
                    "OWN_CAR_AGE": 12.061090818687727,
                    "NAME_INCOME_TYPE_Working": 0,
                    "DAYS_EMPLOYED": -768,
                    "AMT_GOODS_PRICE": 553500,
                    "AMT_CREDIT": 641173.5,
                    "EXT_SOURCE_1_x": 0.8427634659543568,
                    "EXT_SOURCE_2_x": 0.6816988025574287,
                    "EXT_SOURCE_3_x": 0.7544061731797895}

BAD_DICT_ID_CLIENT = {"SK_ID_CURR": 207109,
                      "FLAG_OWN_REALTY": 1,
                      "FLAG_OWN_CAR": 0,
                      "OWN_CAR_AGE": 12.061090818687727,
                      "NAME_INCOME_TYPE_Working": 0,
                      "DAYS_EMPLOYED": -768,
                      "AMT_GOODS_PRICE": 553500,
                      "AMT_CREDIT": 641173.5,
                      "EXT_SOURCE_1_x": 0.8427634659543568,
                      "EXT_SOURCE_2_x": 0.6816988025574287,
                      "EXT_SOURCE_3_x": 0.7544061731797895}

BAD_DICT_FORMAT = {"SK_ID_CURR": 207108,
                   "FLAG_OWN_REALTY": 1,
                   "FLAG_OWN_CAR": 0,
                   "OWN_CAR_AGE": 12.061090818687727,
                   "AMT_CREDIT": 641173.5,
                   "EXT_SOURCE_1_x": 0.8427634659543568,
                   "EXT_SOURCE_2_x": 0.6816988025574287,
                   "EXT_SOURCE_3_x": 0.7544061731797895}


# -------------------------------------------------------------------------------------------------------------------
# Définition des fonctions de test
# -------------------------------------------------------------------------------------------------------------------

def test_matrice():
    """Test si le serveur a renvoyé une matrice correcte"""

    result = requests.get(API_BASE_URL+API_MATRICE, timeout=HTTPS_TIMEOUT)
    assert result.status_code == 200, "Le serveur n'a pas répondu 200 comme attendu"

    # Decode et convertie le JSON format en dictionaire
    dict_obj = json.loads(result.content)

    assert len(dict_obj) == 5, "Le serveur doit renvoyer 5 entrées de 1er niveau"

    # Décodage de la matrice de confusion
    valeurs = dict_obj.get("Matrice de confusion")

    assert valeurs is not None, "Le serveur doit renvoyer une Matrice de confusion"

    valeurs_list = valeurs.split(" - ", 4)

    assert len(valeurs_list) == 4, "La matrice de confusion doit contenir 4 valeurs"

    assert valeurs_list[0][3:].isnumeric(), "La valeur doit être numériques"
    assert valeurs_list[1][3:].isnumeric(), "La valeur doit être numériques"
    assert valeurs_list[2][3:].isnumeric(), "La valeur doit être numériques"
    assert valeurs_list[3][3:].isnumeric(), "La valeur doit être numériques"

    # Vérifie qu'on a bien trouvé l'entrée "Quelques TN"
    some_TN = dict_obj.get("Quelques TN")
    assert some_TN is not None, "Le serveur doit renvoyer des identifiants TN"

    some_TN = some_TN.split(", ")
    assert len(some_TN) == 10, "Quelques TN devrait contenir 10 valeurs"

    # Vérifie qu'on a bien trouvé l'entrée "Quelques FN"
    some_FN = dict_obj.get("Quelques FN")
    assert some_FN is not None, "Le serveur doit renvoyer des identifiants FN"

    some_FN = some_FN.split(", ")
    assert len(some_FN) == 10, "Quelques FN devrait contenir 10 valeurs"

    # Vérifie qu'on a bien trouvé l'entrée "Quelques FP"
    some_FP = dict_obj.get("Quelques FP")
    assert some_FP is not None, "Le serveur doit renvoyer des identifiants FP"

    some_FP = some_FP.split(", ")
    assert len(some_FP) == 10, "Quelques FP devrait contenir 10 valeurs"

    some_TP = dict_obj.get("Quelques TP")
    assert some_TP is not None, "Le serveur doit renvoyer des identifiants TP"

    some_TP = some_TP.split(", ")
    assert len(some_TP) == 10, "Quelques TP devrait contenir 10 valeurs"


# ----------------------------------------------------------

def test_get():
    """Test si le serveur renvoi un client correct"""

    # Test un id qui n'est pas dans la fourchette des id du serveur
    result = requests.get(f"{API_BASE_URL}{API_GET}{BAD_ID_FORMAT}", timeout=HTTPS_TIMEOUT)
    assert result.status_code == 422, "Le serveur n'a pas répondu 422 comme attendu"

    # Test un id inconnu du serveur
    result = requests.get(f"{API_BASE_URL}{API_GET}{BAD_ID_CLIENT}", timeout=HTTPS_TIMEOUT)
    assert result.status_code == 404, "Le serveur n'a pas répondu 404 comme attendu"

    result = requests.get(f"{API_BASE_URL}{API_GET}{GOOD_ID_CLIENT}", timeout=HTTPS_TIMEOUT)
    assert result.status_code == 200, "Le serveur n'a pas répondu 200 comme attendu"

    # Decode et convertie le JSON format en dictionaire
    dict_obj = json.loads(result.content)
    assert len(dict_obj) == 14, "Le client retourné par le serveur doit contenir 14 valeurs"


# ----------------------------------------------------------

def test_post():
    """Test si le serveur renvoi un client correct suite à un post"""

    # Test un dictionnaire dont le format n'est pas conforme
    result = requests.post(f"{API_BASE_URL}{API_POST}", json=BAD_DICT_FORMAT, timeout=HTTPS_TIMEOUT)
    assert result.status_code == 422, "Le serveur n'a pas répondu 422 comme attendu"

    # Test un id inconnu du serveur
    result = requests.post(f"{API_BASE_URL}{API_POST}", json=BAD_DICT_ID_CLIENT, timeout=HTTPS_TIMEOUT)
    assert result.status_code == 404, "Le serveur n'a pas répondu 404 comme attendu"

    result = requests.post(f"{API_BASE_URL}{API_POST}", json=GOOD_DICT_CLIENT, timeout=HTTPS_TIMEOUT)
    assert result.status_code == 200, "Le serveur n'a pas répondu 200 comme attendu"

    # Decode et convertie le JSON format en dictionaire
    dict_obj = json.loads(result.content)
    assert len(dict_obj) == 14, "Le client retourné par le serveur doit contenir 14 valeurs"


# -------------------------------------------------------------------------------------------------------------------
# La fonction MAIN
# Note : ce fichier est prévu pour être exécuté avec pytest : il n'y a donc pas le traditionnel
#   if __name__ == "__main__":
#           main()
# -------------------------------------------------------------------------------------------------------------------

# Test de la matrice de confusion
test_matrice()

# Test d'un GET
test_get()

# Test d'un POST
test_post()
