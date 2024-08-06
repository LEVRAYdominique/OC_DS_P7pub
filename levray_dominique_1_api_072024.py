'''
========================================================================
Projet n°7 - Implémentez un modèle de scoring
Script Python réalisé par Dominique LEVRAY en Juillet/Août 2024
========================================================================
Pour exécuter cette API localement : uvicorn LEVRAY_Dominique_1_API_072024:app --reload
'''
#pylint: disable=line-too-long

# Importation des modules
import  os
import  time
import  re
import  numpy       as      np
import  pandas      as      pd

# FastAPI
from    fastapi     import  FastAPI, Path, HTTPException
from    fastapi     import  UploadFile

# BytesIO
from    io          import BytesIO

# mlflow
import mlflow                                               # MlFlow

print("\n   ")

# Defini quelques constantes des noms de fichier
ZIP_FULL_DATA_FILENAME       = "produced/full_data.zip"         # Fichier de données complet pré-processé
ZIP_TRAIN_DATA_FILENAME      = "produced/train_data.zip"        # Fichier de données d'entrainement partiel (colonnes SK_ID_CURR et TARGET)
ZIP_TEST_DATA_FILENAME       = "produced/test_data.zip"         # Fichier de données de test partiel (colonnes SK_ID_CURR et TARGET)
ZIP_SHAP_IMPORTANCE_FILENAME = "produced/shap_importance.zip"   # Fichier contenant la liste des colonnes importantes (selon SHAP)

# Definition de constante
BEST_THRESHOLD = 0.28                                           # Seuil d'appartenance à la TARGET 1 (lu sur la courbe PR lors de la modelisation)

# Initialisation du model (depuis le fichier sauvegardé)
print("Chargement du modele pré-entraîné...")
model = mlflow.sklearn.load_model("mlflow_model")   

# Chargement des données depuis les 3 fichiers ZIP créés lors de la modélisation
print(f"Lecture des données depuis data/{ZIP_FULL_DATA_FILENAME}")
full_data_df  = pd.read_csv("data/"+ZIP_FULL_DATA_FILENAME,  sep=',', encoding='utf-8',compression='zip')
print(f"\tTaille de full_data_df={full_data_df.shape}")

# print(f"Lecture des données depuis data/{ZIP_TRAIN_DATA_FILENAME}")
# train_data_df = pd.read_csv("data/"+ZIP_TRAIN_DATA_FILENAME, sep=',', encoding='utf-8',compression='zip')
# print(f"\tTaille de train_data_df={train_data_df.shape}")

# print(f"Lecture des données depuis data/{ZIP_TEST_DATA_FILENAME}")
# test_data_df  = pd.read_csv("data/"+ZIP_TEST_DATA_FILENAME,  sep=',', encoding='utf-8',compression='zip')
# print(f"\tTaille de test_data_df={test_data_df.shape}")

# Chargement des données depuis le fichier ZIP sur les colonnes importantes créés lors de la modélisation
print(f"Lecture des données depuis data/{ZIP_SHAP_IMPORTANCE_FILENAME}")
shap_importante_df  = pd.read_csv("data/"+ZIP_SHAP_IMPORTANCE_FILENAME,  sep=',', encoding='utf-8',compression='zip')
print(f"\tTaille de full_data_df={shap_importante_df.shape}")

colonnes_toutes = full_data_df.columns
colonnes_shap   = shap_importante_df['col_name'].to_list
min_SK_ID_CURR  = full_data_df['SK_ID_CURR'].min()
max_SK_ID_CURR  = full_data_df['SK_ID_CURR'].max()

# Initialisation de l'instance FastAPI
app = FastAPI(debug=True)

@app.get("/client/{SK_ID_CURR}") 
def get_client(SK_ID_CURR: int = Path(ge=min_SK_ID_CURR, le=max_SK_ID_CURR)) -> dict:
    '''
    Obtenir les valeurs des 25 features les plus importantes (selon SHAP) pour le client qui a contracté le crédit dont l'id est SK_ID_CURR
    '''
    part_data_df = full_data_df[full_data_df['SK_ID_CURR']==SK_ID_CURR]
    if part_data_df.shape[0]==0:
        raise HTTPException(status_code=404, detail="SK_ID_CURR non trouvé !")
    return {"message" : part_data_df.to_json()}

# def load_model_from_mlflow():
#     '''
#     Défini une fonction pour charger le model sauvegardé par mlflow
#     '''
#     #mlflow_run_id               = "bbe7532cec3740bbbfe9c28c84a78b8f"
#     #run_relative_path_to_model  = "mlflow_model"
#     #model_uri                   = f"runs:/{run_id}/{run_relative_path_to_model}"
#     #model                       = mlflow.sklearn.load_model(f"runs:/{mlflow_run_id}/{run_relative_path_to_model}")
#     return mlflow.sklearn.load_model("mlflow_model")

# @app.post("/predict")                   # Encapsulation des requêtes POST sur "/predict"
# async def predict(file: UploadFile):
#     '''
#     Fonction asynchrone pour renvoyer une prédiction depuis un fichier de donnée fourni via POST
#     '''
#     # Initialise un dataframe avec le fichier .csv envoyé via POST
#     data_file=await file.read()             # Lecture des données binaire du fichier
#     bio      =BytesIO(data_file)            # Lecture en streaming du fichier (sans sauvegarde préalable)
#     data_df  =pd.read_csv(bio)              # Chargement d'un dataframe à partir du fichier lu en streaming
    
#     # Obtention des prédictions
#     y_pred_proba = model.predict_proba(data_df)[:, 1]               # Sous forme de probabilité
#     y_pred       = np.where(y_pred_proba>=BEST_THRESHOLD, 1, 0)     # Calcul de la classe d'appartenance (seuil=best_threshold)
#     #return {"message":f"data_file lu + bio + data_df={data_df.shape} + len(y_pred)={len(y_pred)}"}
    
#     return {"predictions": f"50 premières valeurs{y_pred[:50]}"}

# @app.get("/")                       # Décorateur pour encapsuler les requêtes GET sur "/"
# def hello_world():                        # La fonction à exécuter en réponse au GET
#     '''
#     Fonction pour renvoyer hello_world en GET et vérifier que le code fonctionne
#     '''
#     return {"message":"Hello world !"}   # Renvoi de données sous forme de dictionnaire

