'''
========================================================================
Projet n°7 - Implémentez un modèle de scoring
Script Python réalisé par Dominique LEVRAY en Juillet/Août 2024
========================================================================
Pour exécuter cette API localement : uvicorn levray_dominique_1_api_072024:app --reload
'''
#pylint: disable=line-too-long
#pylint: disable=invalid-name
#pylint: disable=broad-exception-raised
#pylint: disable=trailing-whitespace

# Importation des modules
from pydantic.dataclasses import dataclass      # à utiliser à la place de dataclasses avec FastAPI

# Les modules standards datascience
import  numpy       as      np
import  pandas      as      pd

# FastAPI
from    fastapi     import  FastAPI, Path, HTTPException
#from    fastapi     import  UploadFile

# sklearn.metrics
from    sklearn.metrics             import confusion_matrix

# BytesIO
#from    io          import BytesIO

# mlflow
import mlflow                                               # MlFlow

print("\n   ")

# Defini quelques constantes des noms de fichier
ZIP_TEST_DATA_FILENAME = "test_data.zip"                 # fichier contenant les données de test
MLFLOW_MODEL_FOLDER    = "mlflow_model"                  # fichier contenant le modèle pré-entraîné

# Definition de constante
BEST_THRESHOLD = 0.28                                    # Seuil d'appartenance à la TARGET 1 (lu sur la courbe PR lors de la modelisation)

# Initialisation du model (depuis le fichier sauvegardé)
print("Chargement du modèle pré-entraîné...")
model = mlflow.sklearn.load_model("mlflow_model")

# Chargement des données depuis les 3 fichiers ZIP créés lors de la modélisation
print("Chargement des données de test...")
data_df  = pd.read_csv(ZIP_TEST_DATA_FILENAME,  sep=',', encoding='utf-8',compression='zip')

# Initialise des variables avec les index min et max de SK_ID_CURR
min_SK_ID_CURR  = data_df['SK_ID_CURR'].min()
max_SK_ID_CURR  = data_df['SK_ID_CURR'].max()

# Prépare les données
y = data_df['TARGET']
X = data_df.drop(columns='TARGET')

# Faire la prédiction
y_pred_proba   = model.predict_proba(X)[:, 1]
y_pred         = np.where(y_pred_proba>=BEST_THRESHOLD, 1, 0)
merged_data_df = pd.concat([data_df, pd.DataFrame(y_pred_proba, columns=['y_pred_proba']), pd.DataFrame(y_pred, columns=['y_pred'])], axis=1)

# Calculer la matrice de confusion
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

# Initialisation de l'instance FastAPI
print("\nInitialisation de l'API...")
app = FastAPI(debug=True)

@dataclass
class Client_credit:
    """Class representant un client d'un crédit"""
    SK_ID_CURR:                                     int             # l'index du credit
    FLAG_OWN_REALTY:                                int             # Si le client est propriétaire de son logement
    FLAG_OWN_CAR:                                   int             # Si le client a et est propriétaire d'une voiture
    OWN_CAR_AGE:                                    float           # Age de la voiture
    PAYMENT_RATE:                                   float           # Pourcentage du montant du crédit remboursé annuellement [pré-processing]
    NAME_INCOME_TYPE_Working:                       bool            # True si les revennus du client proviennent d'un salaire
    NAME_INCOME_TYPE_Commercialassociate:           bool            # True si les revennus du client sont des commissions (commerce)
    NAME_EDUCATION_TYPE_Highereducation:            bool            # True si le client a fait des études supérieurs
    NAME_EDUCATION_TYPE_Secondarysecondaryspecial:  bool            # True si le client a fini les études secondaires
    NAME_FAMILY_STATUS_Married:                     bool            # True si le client est marié
    NAME_FAMILY_STATUS_Singlenotmarried:            bool            # True si le client est célibataire
    PRED_PROBA:                                     float = 0       # probabilité que le client ait des retards de paiement
    PRED_TARGET:                                    int = 0         # TARGET predite et calculé à partir de PRED_PROBA
    TARGET:                                         bool = 0        # 0 si le client a des retards de paiement => crédit à rejeter

@dataclass
class Client_new_credit:
    """Class representant un client pour un nouveau crédit (même structure que Client_credit mais sans TARGET ni prédiction)"""
    SK_ID_CURR:                                     int             # l'index du crédit
    FLAG_OWN_REALTY:                                int             # Si le client est propriétaire de son logement
    FLAG_OWN_CAR:                                   int             # Si le client a et est propriétaire d'une voiture
    OWN_CAR_AGE:                                    float           # Age de la voiture
    PAYMENT_RATE:                                   float           # Pourcentage du montant du crédit remboursé annuellement [pré-processing]
    NAME_INCOME_TYPE_Working:                       bool            # True si les revennus du client proviennent d'un salaire
    NAME_INCOME_TYPE_Commercialassociate:           bool            # True si les revennus du client sont des commissions (commerce)
    NAME_EDUCATION_TYPE_Highereducation:            bool            # True si le client a fait des études supérieurs
    NAME_EDUCATION_TYPE_Secondarysecondaryspecial:  bool            # True si le client a fini les études secondaires
    NAME_FAMILY_STATUS_Married:                     bool            # True si le client est marié
    NAME_FAMILY_STATUS_Singlenotmarried:            bool            # True si le client est célibataire

#--------------------------------------------------------------------------------------------------------------------------------

def Client_credit_from_data(client_data):
    """Obtient un Client_credit à partir d'un SK_ID_CURR"""

    the_client=Client_credit(SK_ID_CURR                                      = client_data['SK_ID_CURR'],
                             FLAG_OWN_REALTY                                 = client_data['FLAG_OWN_REALTY'],
                             FLAG_OWN_CAR                                    = client_data['FLAG_OWN_CAR'],
                             OWN_CAR_AGE                                     = client_data['OWN_CAR_AGE'],
                             PAYMENT_RATE                                    = client_data['PAYMENT_RATE'],
                             NAME_INCOME_TYPE_Working                        = client_data['NAME_INCOME_TYPE_Working'],
                             NAME_INCOME_TYPE_Commercialassociate            = client_data['NAME_INCOME_TYPE_Commercialassociate'],
                             NAME_EDUCATION_TYPE_Highereducation             = client_data['NAME_EDUCATION_TYPE_Highereducation'],
                             NAME_EDUCATION_TYPE_Secondarysecondaryspecial   = client_data['NAME_EDUCATION_TYPE_Secondarysecondaryspecial'],
                             NAME_FAMILY_STATUS_Married                      = client_data['NAME_FAMILY_STATUS_Married'],
                             NAME_FAMILY_STATUS_Singlenotmarried             = client_data['NAME_FAMILY_STATUS_Singlenotmarried'],
                             PRED_PROBA                                      = client_data['y_pred_proba'],
                             PRED_TARGET                                     = client_data['y_pred'],
                             TARGET                                          = client_data['TARGET']
                            )
    return the_client

#--------------------------------------------------------------------------------------------------------------------------------

@app.get("/get_client/{SK_ID_CURR}")
def get_client_by_ID(SK_ID_CURR: int = Path(ge=min_SK_ID_CURR, le=max_SK_ID_CURR)) -> Client_credit:
    '''Obtenir quelques valeurs importantes pour le client qui a contracté le crédit dont l'id est SK_ID_CURR'''
    
    # Commence par vérifier que l'ID est valide !
    part_data_df = merged_data_df[merged_data_df['SK_ID_CURR']==SK_ID_CURR]
    if part_data_df.shape[0]==0:
        raise HTTPException(status_code=404, detail="SK_ID_CURR non trouvé !")

    # Renvoi les informations pour le client en question
    return Client_credit_from_data(part_data_df)

#--------------------------------------------------------------------------------------------------------------------------------

def Client_credit_to_new_data(client_data: Client_new_credit):
    """Obtient un dataframe d'une ligne à partir de data_df et d'un Client_credit contenant des informations modifiées"""

    # Construit un dataframe d'une ligne
    new_data_df = merged_data_df[merged_data_df['SK_ID_CURR']==client_data.SK_ID_CURR].copy()   # Copy car on va modifier la donnée

    # Transfert les nouvelles données dans le dataframe
    new_data_df['FLAG_OWN_REALTY']                               = client_data.FLAG_OWN_REALTY
    new_data_df['FLAG_OWN_CAR']                                  = client_data.FLAG_OWN_CAR
    new_data_df['OWN_CAR_AGE']                                   = client_data.OWN_CAR_AGE
    new_data_df['PAYMENT_RATE']                                  = client_data.PAYMENT_RATE
    new_data_df['NAME_INCOME_TYPE_Working']                      = client_data.NAME_INCOME_TYPE_Working
    new_data_df['NAME_INCOME_TYPE_Commercialassociate']          = client_data.NAME_INCOME_TYPE_Commercialassociate
    new_data_df['NAME_EDUCATION_TYPE_Highereducation']           = client_data.NAME_EDUCATION_TYPE_Highereducation
    new_data_df['NAME_EDUCATION_TYPE_Secondarysecondaryspecial'] = client_data.NAME_EDUCATION_TYPE_Secondarysecondaryspecial
    new_data_df['NAME_FAMILY_STATUS_Married']                    = client_data.NAME_FAMILY_STATUS_Married
    new_data_df['NAME_FAMILY_STATUS_Singlenotmarried']           = client_data.NAME_FAMILY_STATUS_Singlenotmarried

    # il faut recalculer la prédiction pour le cas où des valeurs aient changées
    new_X       = new_data_df.drop(columns=['TARGET', 'y_pred_proba', 'y_pred'])
    new_y_proba = model.predict_proba(new_X)[:, 1][0]
    new_y_pred  = int(np.where(new_y_proba>=BEST_THRESHOLD, 1, 0))

    # et transférer ces nouvelles prédictions dans le dataframe d'une ligne
    new_data_df['y_pred_proba'] = new_y_proba
    new_data_df['y_pred']       = new_y_pred

    return new_data_df

#--------------------------------------------------------------------------------------------------------------------------------

@app.post("/post_client/{Client_new_credit}")
def calcul_nouveau_credit(new_client: Client_new_credit) -> Client_credit:
    '''Recalculer les prédictions pour de nouvelles valeurs d'un client/crédit existant'''

    # Commencer par vérifier que l'ID est valide !
    part_data_df = data_df[data_df['SK_ID_CURR']==new_client.SK_ID_CURR]
    if part_data_df.shape[0]==0:
        raise HTTPException(status_code=404, detail="SK_ID_CURR non trouvé !")

    return Client_credit_from_data(Client_credit_to_new_data(new_client))

#--------------------------------------------------------------------------------------------------------------------------------

@app.get("/matrice_confusion") 
def matrice_confusion() -> dict:
    '''Obtenir la matrice de confusion pour toutes les données de test + quelques identifiant choisi aléatoirement pour chaque catégorie'''
    return {"Matrice de confusion": f"TN={tn} - FN={fn} - FP={fp} - TP={tp}",
            "Quelques TN":          ", ".join(map(str, merged_data_df[(merged_data_df['y_pred']==1) & (merged_data_df['TARGET']==0)]['SK_ID_CURR'].sample(10).to_list())),
            "Quelques FN":          ", ".join(map(str, merged_data_df[(merged_data_df['y_pred']==0) & (merged_data_df['TARGET']==1)]['SK_ID_CURR'].sample(10).to_list())),
            "Quelques FP":          ", ".join(map(str, merged_data_df[(merged_data_df['y_pred']==0) & (merged_data_df['TARGET']==0)]['SK_ID_CURR'].sample(10).to_list())),
            "Quelques TP":          ", ".join(map(str, merged_data_df[(merged_data_df['y_pred']==1) & (merged_data_df['TARGET']==1)]['SK_ID_CURR'].sample(10).to_list()))
           }
