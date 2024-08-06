'''
========================================================================
Projet n°7 - Implémentez un modèle de scoring
Script Python réalisé par Dominique LEVRAY en Juillet/Août 2024
========================================================================
Module pour retraiter tous les petits fichiers de données en un
dataframe unique exploitable par l'API 
+ faire le preprocessing des données en :
- corrigeant des problèmes de valeurs vides ou abérrantes
- créant des valeurs métiers
- créant des variables polynomiales
'''
#pylint: disable=line-too-long
#pylint: disable=f-string-without-interpolation
#pylint: disable=broad-exception-caught

import  re
import  numpy                       as np
import  pandas                      as pd
import  category_encoders           as ce

from    sklearn.preprocessing       import LabelEncoder
from    sklearn.preprocessing       import PolynomialFeatures

###############################################################################################################
# Définir quelques fonctions utilitaires
###############################################################################################################

def read_csv_file(filename):
    """ 
     Cree un dataframe depuis un fichier .csv
    """
    print(f"\nLecture du fichier csv : {filename}")
    try:
        data_df = pd.read_csv(filename,             # Nom du fichier
                            sep      = ',',         # Séparateur virgule
                            encoding = 'utf-8',     # Encodage UTF-8
                            )
    except Exception:
        data_df = pd.read_csv(filename,             # Nom du fichier
                            sep      = ',',         # Séparateur virgule
                            encoding = 'iso-8859-1' # Change d'encodage
                            )

    print(f"\tLe fichier {filename} contient {data_df.shape[0]} lignes et {data_df.shape[1]} colonnes.")
    return data_df

###############################################################################################################

def write_csv_zip_file(filename, data_df):
    """ 
     Cree un fichier .csv depuis un dataframe
    """
    print(f"Ecriture du fichier csv : data/{filename}\n")
    data_df.to_csv(filename,                # Nom du fichier
                   sep=',',                 # Séparateur virgule
                   encoding = 'utf-8',      # Encodage UTF-8
                   index=False,             # N'inclus pas l'index
                   compression='zip'
                  )

###############################################################################################################

def supp_data_inutiles(df, var_id, list_id, nom_df, nom_liste):
    ''' 
    Supprimer les individus d'un dataframe sur un identifiant contenu dans une variable qui n'est pas présent dans une liste 
    '''
    print(f"\tSupprime les individus de {nom_df} pour lesquels l'identifiant {var_id} n'est pas présent dans {nom_liste}")
    anc_nbr_lignes = df.shape[0]
    df.drop(df[~df[var_id].isin(list_id)].index, inplace=True)
    print(f"\t\t{nom_df} contient désormais {df.shape[0]} au lieu de {anc_nbr_lignes} lignes soit {df.shape[0]-anc_nbr_lignes} lignes")

###############################################################################################################

def prepare_bureau_balance(bureau_balance_df):
    '''
    Retraite les données de bureau_balance avant fusion
    '''
    print("\n\tPréparation de bureau_balance")

    print("\t\tbureau_balance : OneHotEncodage de la variable STATUS")
    encoder           = ce.OneHotEncoder(cols=['STATUS'], use_cat_names=True)   # Instancie un category_encoders "OneHotEncoder"
    bureau_balance_df = encoder.fit_transform(bureau_balance_df)                # Réalise la transformation

    print("\t\tbureau_balance : Compte le nombre d'occurences par STATUS")
    col       = encoder.get_feature_names_out()[2:]                             # Récupère les noms des colonnes de status
    dict_agg  = {col[i]: 'sum' for i in range(0, len(col))}                     # Construit un dictionnaire d'aggregation (toutes les colonnes en 'sum')
    dict_agg['MONTHS_BALANCE']='min'                                            # Ajoute l'agrégation pour MONTHS_BALANCE

    # Compte le nombre d'occurence par STATUS
    print("\t\tbureau_balance : Groupe les occurences mensuels par crédit")
    bureau_balance_df = bureau_balance_df.groupby('SK_ID_BUREAU', as_index=False).agg(dict_agg)
    bureau_balance_df = bureau_balance_df.reset_index(drop=True)                # Reindex le résultat pour la suite

    # Crée la colonne Incident_OuiNon (Commence en additionnant les valeurs des colonnes STATUS_1 à STATUS_5)
    print("\t\tbureau_balance : Transforme les nombres d'incidents en une colonne binaire")
    bureau_balance_df['INCIDENTS']=bureau_balance_df['STATUS_1']+bureau_balance_df['STATUS_2']+bureau_balance_df['STATUS_3']+bureau_balance_df['STATUS_4']+bureau_balance_df['STATUS_5']
    bureau_balance_df.loc[bureau_balance_df['INCIDENTS']>0, 'INCIDENTS']=1  # Puis passe en 0/1

    # Ne conserve que les colonnes SK_ID_BUREAU et INCIDENTS
    bureau_balance_df=bureau_balance_df[['SK_ID_BUREAU', 'MONTHS_BALANCE', 'INCIDENTS']]

    # Affiche un aperçu du résultat de la transformation
    print("\t\tbureau_balance : contient désormais {bureau_balance_df.shape[0]} lignes et {bureau_balance_df.shape[1]} colonnes.")

    return bureau_balance_df

###############################################################################################################

def prepare_bureau(bureau_df, bureau_balance_df):
    '''
    Retraite les données de bureau avant fusion
    '''
    print("\n\tPréparation de bureau")

    # Supprime 2 colonnes contenant trop de valeurs vides
    print("\t\tbureau : Supprime les 2 colonnes contenant trop de valeurs vides AMT_ANNUITY et AMT_CREDIT_MAX_OVERDUE")
    del bureau_df['AMT_ANNUITY']
    del bureau_df['AMT_CREDIT_MAX_OVERDUE']

    # Remplace les valeurs vide des autres colonnes par 0
    print("\t\tbureau : Remplace les valeurs vides des autres colonnes par 0")
    bureau_df.replace(np.nan, 0, inplace=True)

    # Joins les lignes de bureau_balance_df aux lignes de bureau_df
    print("\t\tbureau : Fusionne bureau_balance_df et bureau_df")
    bureau_df = pd.merge(bureau_df, bureau_balance_df, left_on="SK_ID_BUREAU", right_on="SK_ID_BUREAU", how="outer")

    # Supprime la colonne SK_ID_BUREAU qui ne servira plus
    del bureau_df['SK_ID_BUREAU']

    # Calcul la medianne de MONTHS_BALANCE dans le dataframe bureau_balance_df
    median_months_balance = np.median(bureau_balance_df['MONTHS_BALANCE'])

    # Les crédits de bureau_df non connus dans bureau_balance_df se retrouve avec des NaN dans les 2 nouvelles colonnes INCIDENTS et MONTHS_BALANCE
    bureau_df.loc[bureau_df['INCIDENTS'].isnull(),      'INCIDENTS']=0                          # Remplacer les NaN de INCIDENTS par des 0 (car on a pas connaissance d'incidents pour ces crédits)
    bureau_df.loc[bureau_df['MONTHS_BALANCE'].isnull(), 'MONTHS_BALANCE']=median_months_balance # Remplacer les NaN de MONTHS_BALANCE par la mediane des MONTHS_BALANCE

    print("\t\tbureau : Suppression des crédits dans les monnaies autre que currency 1")
    anc_nbr_lignes = bureau_df.shape[0]
    bureau_df.drop(bureau_df[bureau_df['CREDIT_CURRENCY'].isin(['currency 2', 'currency 3', 'currency 4'])].index, inplace=True)
    print(f"\t\tbureau : bureau_df contient désormais {bureau_df.shape[0]} lignes au lieu de {anc_nbr_lignes} lignes")

    print("\t\tbureau : Suppression de la colonne CREDIT_CURRENCY")
    del bureau_df['CREDIT_CURRENCY']

    # Compte le nombre de crédits clos et le nombre de crédit clos pour lesquels au moins un incident de paiemment est survennus
    bureau_clos_df = bureau_df[bureau_df['CREDIT_ACTIVE']=='Closed'].groupby('SK_ID_CURR', as_index=False).agg(CREDITS_CLOS=('SK_ID_CURR', 'count'), INCIDENTS_CREDITS_CLOS=('INCIDENTS', 'max')).reset_index(drop=True)

    # Compte le nombre de crédits actif et le nombre de crédit actif pour lesquels au moins un incident de paiemment est survennus
    bureau_actifs_df = bureau_df[bureau_df['CREDIT_ACTIVE']=='Active'].groupby('SK_ID_CURR', as_index=False).agg(CREDITS_ACTIFS=('SK_ID_CURR', 'count'), INCIDENTS_CREDITS_ACTIFS=('INCIDENTS', 'max')).reset_index(drop=True)

    # Joins les deux dataframe clos et actifs
    print("\t\tbureau : Agrege les nombres de credits clos et actifs et les nombre d'incidents")
    bureau_incidents_df = pd.merge(bureau_clos_df, bureau_actifs_df, left_on="SK_ID_CURR", right_on="SK_ID_CURR", how="outer")

    # Supprime les dataframe devennus inutiles => force le garbage collector
    del bureau_clos_df
    del bureau_actifs_df

    # Remplace les valeurs NAN par des 0 (il se peut qu'il n'y ait qu'un type de crédit (actif ou clos) pour ID_CURR)
    print("\t\tbureau : Remplace les valeurs NAN par des 0")
    for col in ['CREDITS_CLOS', 'INCIDENTS_CREDITS_CLOS', 'CREDITS_ACTIFS', 'INCIDENTS_CREDITS_ACTIFS']:
        bureau_incidents_df.loc[bureau_incidents_df[col].isnull(), col]=0

    print("\t\tbureau : Suppression des crédits clos pour n'agréger que les informations des crédits encore actifs")
    bureau_df.drop(bureau_df[bureau_df['CREDIT_ACTIVE']=='Closed'].index, inplace=True)

    aggregations = {
        'DAYS_CREDIT'           : 'median',                     # Ancienneté du précédent crédit (en jours)
        'DAYS_CREDIT_ENDDATE'   : 'median',                     # Durée restante du crédit CB (en jours)
        'DAYS_CREDIT_UPDATE'    : 'median',                     # Ancienneté des informations (en jours)
        'CREDIT_DAY_OVERDUE'    : 'median',                     # Nombre de jours de retard sur le crédit CB
        'DAYS_ENDDATE_FACT'     : 'median',                     # Nombre de jours depuis la cloture du crédit CB
        'AMT_CREDIT_SUM'        : 'sum',                        # Montant actuel de l'encourt crédit
        'AMT_CREDIT_SUM_DEBT'   : 'sum',                        # Montant débiteur actuel sur les crédits en cours
        'AMT_CREDIT_SUM_OVERDUE': 'sum',                        # Montant débiteur actuel sur l'encourt crédit
        'AMT_CREDIT_SUM_LIMIT'  : 'max',                        # Limite de crédit actuelle de la CB
        'CNT_CREDIT_PROLONG'    : 'sum',                        # Combien de fois le crédit a-t-il été prolongé
        'MONTHS_BALANCE'        : 'median',                     # Medianne des durées de crédit en mois
        #'CREDIT_TYPE'           : lambda x: x.mode().iloc[0]   # Comme on agrege l'historique de plusieurs types de crédits : on ne garde pas cette information
    }

    bureau_num_df  = bureau_df[bureau_df['CREDIT_ACTIVE']=='Active'].groupby('SK_ID_CURR', as_index=False).agg(aggregations)
    bureau_num_df  = bureau_num_df.reset_index(drop=True)       # Reindex le résultat pour la suite

    # Finalement on joins bureau_num_df et bureau_incidents_df
    bureau_df = pd.merge(bureau_num_df, bureau_incidents_df, left_on="SK_ID_CURR", right_on="SK_ID_CURR", how="outer")

    # Supprime les dataframe devennus inutiles => force le garbage collector
    del bureau_num_df
    del bureau_incidents_df

    # Remplace les valeurs NAN par des 0 (il se peut qu'il n'y ait pas de crédit actif pour ID_CURR)
    for col in bureau_df.columns:
        bureau_df.loc[bureau_df[col].isnull(), col]=0

    # Renomme certaines colonnes en préparation de la future fusion avec le fichier principal
    print("\t\tbureau : Renomme quelques colonnes ...")
    bureau_df.rename(columns={'CREDITS_ACTIFS':           'B_CREDITS_ACTIFS'},           inplace=True)
    bureau_df.rename(columns={'INCIDENTS_CREDITS_ACTIFS': 'B_INCIDENTS_CREDITS_ACTIFS'}, inplace=True)
    bureau_df.rename(columns={'MONTHS_BALANCE':           'B_MONTHS_BALANCE'},           inplace=True)

    # Affiche un aperçu du résultat de la transformation
    print(f"\t\tbureau : Après traitement, bureau_df contient {bureau_df.shape[0]} lignes et {bureau_df.shape[1]} colonnes.")
    return bureau_df

###############################################################################################################

def prepare_credit_card_balance(credit_card_balance_df):
    '''
    Retraite les données de credit_card_balance avant fusion
    '''
    print("\n\tPréparation de credit_card_balance")

    # Plusieurs variables ont des valeurs vides qu'on remplace par 0
    print("\t\tcredit_card_balance : Remplace les valeurs vides par 0")
    credit_card_balance_df.replace(np.nan, 0, inplace=True)

    # Retraite les 2 colonnes SK_DPD et SK_DPD_DEF avant agrégation en binaire oui/non
    print("\t\tcredit_card_balance : Retraite les 2 colonnes SK_DPD et SK_DPD_DEF avant agrégation en binaire oui/non")
    credit_card_balance_df.loc[credit_card_balance_df['SK_DPD']>0,     'SK_DPD'] = 1
    credit_card_balance_df.loc[credit_card_balance_df['SK_DPD_DEF']>0, 'SK_DPD_DEF'] = 1

    # Prépare l'agrégation
    agg_data_cc_bal = {
        'SK_ID_PREV'                    : 'count',                      # Lien avec previous_application => abandonné au profit du lien direct avec application_train : est transformé en nombre de crédit actif
        'MONTHS_BALANCE'                : 'median',
        'AMT_BALANCE'                   : 'mean',
        'AMT_CREDIT_LIMIT_ACTUAL'       : 'median',
        'AMT_DRAWINGS_ATM_CURRENT'      : 'mean',
        'AMT_DRAWINGS_CURRENT'          : 'mean',
        'AMT_DRAWINGS_OTHER_CURRENT'    : 'mean',
        'AMT_DRAWINGS_POS_CURRENT'      : 'mean',
        'AMT_INST_MIN_REGULARITY'       : 'median',
        'AMT_PAYMENT_CURRENT'           : 'median',
        'AMT_PAYMENT_TOTAL_CURRENT'     : 'median',
        'AMT_RECEIVABLE_PRINCIPAL'      : 'mean',
        'AMT_RECIVABLE'                 : 'mean',
        'AMT_TOTAL_RECEIVABLE'          : 'median',
        'CNT_DRAWINGS_ATM_CURRENT'      : 'sum',
        'CNT_DRAWINGS_CURRENT'          : 'sum',
        'CNT_DRAWINGS_OTHER_CURRENT'    : 'sum',
        'CNT_DRAWINGS_POS_CURRENT'      : 'sum',
        'CNT_INSTALMENT_MATURE_CUM'     : 'sum',
        #'NAME_CONTRACT_STATUS'          : lambda x: (x == 'Active').sum(), # Ne prend pas en compte les crédits annulés => Reviens au même que de compter SK_ID_PREV
        'SK_DPD'                        : 'median',                     # Nombre d'incidents
        'SK_DPD_DEF'                    : 'median'                      # Nombre d'incidents avec tolerance
    }

    # Fait l'agrégation
    print("\t\tcredit_card_balance : Fait l'agrégation")
    anc_nbr_lignes         = credit_card_balance_df.shape[0]
    credit_card_balance_df = credit_card_balance_df.groupby('SK_ID_CURR', as_index=False).agg(agg_data_cc_bal)
    credit_card_balance_df = credit_card_balance_df.reset_index(drop=True)       # Reindex le résultat pour la suite

    # Transforme les 2 colonnes SK_DPD et SK_DPD_DEF en une seul colonne incident
    print("\t\tcredit_card_balance : Transforme les 2 colonnes SK_DPD et SK_DPD_DEF en une seul colonne incident et renommage de SK_ID_PREV")
    credit_card_balance_df['CCB_INCIDENTS_CREDITS_ACTIFS'] = credit_card_balance_df['SK_DPD'] + credit_card_balance_df['SK_DPD_DEF']
    credit_card_balance_df.loc[credit_card_balance_df['CCB_INCIDENTS_CREDITS_ACTIFS']>0, 'CCB_INCIDENTS_CREDITS_ACTIFS'] = 1    # Passe en binaire oui/non
    credit_card_balance_df.rename(columns={'SK_ID_PREV': 'CCB_CREDITS_ACTIFS'}, inplace=True)
    del credit_card_balance_df['SK_DPD']
    del credit_card_balance_df['SK_DPD_DEF']

    # Renomme MONTHS_BALANCE en préparation de la future fusion avec le fichier principal
    credit_card_balance_df.rename(columns={'MONTHS_BALANCE': 'CCB_MONTHS_BALANCE'}, inplace=True)

    print(f"\t\tcredit_card_balance : Après agrégation, credit_card_balance_df contient {credit_card_balance_df.shape[0]} lignes au lieu de {anc_nbr_lignes} lignes")

    return credit_card_balance_df

###############################################################################################################

def prepare_pos_cash_balance(pos_cash_balance_df):
    '''
    Retraite les données de pos_cash_balance avant fusion
    '''
    print("\n\tPréparation de pos_cash_balance")

    # CNT_INSTALMENT et CNT_INSTALMENT_FUTURE ont quelques valeurs vides qu'on remplace par 0
    print("\t\tpos_cash_balance : Remplace les valeurs vides par 0")
    pos_cash_balance_df.replace(np.nan, 0, inplace=True)

    # Retraite les 2 colonnes SK_DPD et SK_DPD_DEF avant agrégation en binaire oui/non
    print("\t\tpos_cash_balance : Retraite les 2 colonnes SK_DPD et SK_DPD_DEF avant agrégation en binaire oui/non")
    pos_cash_balance_df.loc[pos_cash_balance_df['SK_DPD']>0,     'SK_DPD']     = 1
    pos_cash_balance_df.loc[pos_cash_balance_df['SK_DPD_DEF']>0, 'SK_DPD_DEF'] = 1

    # Prépare l'agrégation
    agg_data_pos_cash = {
        'SK_ID_PREV'            : 'count',                              # Lien avec previous_application => abandonné au profit du lien direct avec application_train : est transformé en nombre de crédit actif
        'MONTHS_BALANCE'        : 'median',
        'CNT_INSTALMENT'        : 'median',
        'CNT_INSTALMENT_FUTURE' : 'median',
        #'NAME_CONTRACT_STATUS'  : lambda x: (x != 'Canceled').sum(),    # Ne prend pas en compte les crédits annulés => Reviens au même que de compter SK_ID_PREV
        'SK_DPD'                : 'sum',                                # Nombre d'incidents
        'SK_DPD_DEF'            : 'sum'                                 # Nombre d'incidents avec tolerance
    }

    # Fait l'agrégation
    print("\t\tpos_cash_balance : Fait l'agrégation")
    anc_nbr_lignes      = pos_cash_balance_df.shape[0]
    pos_cash_balance_df = pos_cash_balance_df.groupby('SK_ID_CURR', as_index=False).agg(agg_data_pos_cash)
    pos_cash_balance_df = pos_cash_balance_df.reset_index(drop=True)       # Reindex le résultat pour la suite

    # Transforme les 2 colonnes SK_DPD et SK_DPD_DEF en une seul colonne incident
    print("\t\tpos_cash_balance : Transforme les 2 colonnes SK_DPD et SK_DPD_DEF en une seul colonne incident et renommage de SK_ID_PREV")
    pos_cash_balance_df['PCB_INCIDENTS_CREDITS_ACTIFS'] = pos_cash_balance_df['SK_DPD'] + pos_cash_balance_df['SK_DPD_DEF']
    pos_cash_balance_df.loc[pos_cash_balance_df['PCB_INCIDENTS_CREDITS_ACTIFS']>0, 'PCB_INCIDENTS_CREDITS_ACTIFS'] = 1    # Passe en binaire oui/non
    del pos_cash_balance_df['SK_DPD']
    del pos_cash_balance_df['SK_DPD_DEF']
    pos_cash_balance_df.rename(columns={'SK_ID_PREV': 'PCB_CREDITS_ACTIFS'}, inplace=True)

    # Renomme MONTHS_BALANCE en préparation de la future fusion avec le fichier principal
    pos_cash_balance_df.rename(columns={'MONTHS_BALANCE': 'PCB_MONTHS_BALANCE'}, inplace=True)

    print(f"\t\tpos_cash_balance : Après agrégation, pos_cash_balance_df contient {pos_cash_balance_df.shape[0]} lignes au lieu de {anc_nbr_lignes} lignes")

    return pos_cash_balance_df

###############################################################################################################

def preprocessing():
    '''
    La fonction de preprocessing global

    En sortie :
        Renvoi un dataframe complet avec toutes les données retraitées
    '''

    print("\n    ")

    # application_train
    app_train_df = read_csv_file("data/sources/application_train.csv")

    # bureau
    bureau_df = read_csv_file("data/sources/bureau.csv")
    supp_data_inutiles(bureau_df, 'SK_ID_CURR', app_train_df['SK_ID_CURR'].tolist(), "bureau.csv", "application_train.csv")

    # bureau_balance_df
    bureau_balance_df = read_csv_file("data/sources/bureau_balance.csv")
    supp_data_inutiles(bureau_balance_df, 'SK_ID_BUREAU', bureau_df['SK_ID_BUREAU'].tolist(), "bureau_balance.csv", "bureau.csv")
    bureau_balance_df = prepare_bureau_balance(bureau_balance_df)
    bureau_df         = prepare_bureau(bureau_df, bureau_balance_df)
    # Supprime bureau_balance_df qui ne servira plus => force le garbage collector
    del bureau_balance_df

    # previous_application => Note : On ne charge ce fichier que pour supprimer les données inutiles de pos_cash_balance et credit_card_balance
    previous_application_df = read_csv_file("data/sources/previous_application.csv")
    supp_data_inutiles(previous_application_df, 'SK_ID_CURR', app_train_df['SK_ID_CURR'].tolist(), "previous_application.csv", "application_train.csv")

    # pos_cash_balance
    pos_cash_balance_df = read_csv_file("data/sources/pos_cash_balance.csv")
    supp_data_inutiles(pos_cash_balance_df, 'SK_ID_CURR', app_train_df['SK_ID_CURR'].tolist(), "pos_cash_balance.csv", "application_train.csv")
    supp_data_inutiles(pos_cash_balance_df, 'SK_ID_PREV', previous_application_df['SK_ID_PREV'].tolist(), "pos_cash_balance.csv", "previous_application.csv")
    pos_cash_balance_df = prepare_pos_cash_balance(pos_cash_balance_df)

    # credit_card_balance
    credit_card_balance_df = read_csv_file("data/sources/credit_card_balance.csv")
    supp_data_inutiles(credit_card_balance_df, 'SK_ID_PREV', previous_application_df['SK_ID_PREV'].tolist(), "credit_card_balance.csv", "previous_application.csv")
    credit_card_balance_df = prepare_credit_card_balance(credit_card_balance_df)

    # Supprime previous_application_df qui ne servira plus => force le garbage collector
    del previous_application_df

    print(f"\nFusion des fichiers bureau, credit_card_balance et pos_cash")

    # Fusionne les dataframes des fichiers secondaires
    temp_df = pd.merge(bureau_df, credit_card_balance_df, left_on="SK_ID_CURR", right_on="SK_ID_CURR", how="outer")
    temp_df = pd.merge(temp_df,   pos_cash_balance_df,    left_on="SK_ID_CURR", right_on="SK_ID_CURR", how="outer")

    # Additionne les colonnes B_CREDITS_ACTIFS, CCB_CREDITS_ACTIFS et PCB_CREDITS_ACTIFS
    temp_df['CREDITS_ACTIFS'] = temp_df['B_CREDITS_ACTIFS'] + temp_df['CCB_CREDITS_ACTIFS'] + temp_df['PCB_CREDITS_ACTIFS']
    del temp_df['B_CREDITS_ACTIFS']
    del temp_df['CCB_CREDITS_ACTIFS']
    del temp_df['PCB_CREDITS_ACTIFS']

    # Additionne les colonnes B_INCIDENTS_CREDITS_ACTIFS, CCB_INCIDENTS_CREDITS_ACTIFS et PCB_INCIDENTS_CREDITS_ACTIFS
    temp_df['INCIDENTS_CREDITS_ACTIFS'] = temp_df['B_INCIDENTS_CREDITS_ACTIFS'] + temp_df['CCB_INCIDENTS_CREDITS_ACTIFS'] + temp_df['PCB_INCIDENTS_CREDITS_ACTIFS']
    temp_df.loc[temp_df['INCIDENTS_CREDITS_ACTIFS']>0, 'INCIDENTS_CREDITS_ACTIFS'] = 1  # Passe en binaire oui/non
    del temp_df['B_INCIDENTS_CREDITS_ACTIFS']
    del temp_df['CCB_INCIDENTS_CREDITS_ACTIFS']
    del temp_df['PCB_INCIDENTS_CREDITS_ACTIFS']

    # Fait la moyenne des B_MONTHS_BALANCE, CCB_MONTHS_BALANCE et PCB_MONTHS_BALANCE
    temp_df['MONTHS_BALANCE'] = (temp_df['B_MONTHS_BALANCE'] + temp_df['CCB_MONTHS_BALANCE'] + temp_df['PCB_MONTHS_BALANCE'])/3
    del temp_df['B_MONTHS_BALANCE']
    del temp_df['CCB_MONTHS_BALANCE']
    del temp_df['PCB_MONTHS_BALANCE']

    # Remplace les valeurs vides par 0
    print("\tRemplace les valeurs vides par 0")
    temp_df.replace(np.nan, 0, inplace=True)

    # Renomme toutes les colonnes pour les préfixées avec un "_SEC_" afin de pouvoir les reconnaitre par la suite
    for col in temp_df.columns:
        if col != 'SK_ID_CURR':
            temp_df.rename(columns={col : "_SEC_" + col}, inplace=True)

    # Fusionne avec le fichier principal
    app_train_df = pd.merge(app_train_df, temp_df, left_on="SK_ID_CURR", right_on="SK_ID_CURR", how="outer")

    # Supprime bureau_df, credit_card_balance_df et pos_cash_balance_df et temp_df qui ne serviront plus => force le garbage collector
    del bureau_df
    del credit_card_balance_df
    del pos_cash_balance_df
    del temp_df

    print(f"\tAprès fusion des fichiers app_train_df contient {app_train_df.shape[0]} lignes et {app_train_df.shape[1]} colonnes")

    print("\t\tTraitement des anomalies sur la variable DAYS_EMPLOYED")
    app_train_df['DAYS_EMPLOYED_ANOM'] = app_train_df["DAYS_EMPLOYED"]==365243                      # Creation d'une colonne indiquant une anomalie
    app_train_df['DAYS_EMPLOYED']      = app_train_df['DAYS_EMPLOYED'].replace({365243: np.nan})    # Remplace les valeurs en anomalie par nan

    print("\t\tTraitement des valeurs manquantes, aberrantes ou mal renseignées")
    # Remplace toutes les valeurs XNA et XAP en NaN dans plusieursvariables catégorielles (erreur de saisie ?)
    app_train_df.replace("XNA", np.nan,inplace=True)
    app_train_df.replace("XAP", np.nan,inplace=True)

    # Chercher les colonnes numériques qui contiennent des valeurs manquantes
    numeric_col_with_nan = list(set(app_train_df.columns[app_train_df.isnull().sum()!=0].tolist()) & set(app_train_df.select_dtypes(include=np.number).columns.tolist()))

    # Remplir les valeurs vides par la moyenne
    for col in numeric_col_with_nan:
        mean_val          = app_train_df[col].mean()
        app_train_df[col] = app_train_df[col].fillna(mean_val)

    # Chercher les colonnes textes qui contiennent des valeurs manquantes
    object_col_with_nan = list(set(app_train_df.columns[app_train_df.isnull().sum()!=0].tolist()) & set(app_train_df.select_dtypes(include=object).columns.tolist()))

    # Remplace toutes les valeurs NaN par "" pour toutes les colonnes du dataframe
    app_train_df = app_train_df.fillna("")

    #Remplacement des valeurs catégorielles égale à "" par la valeur la plus fréquente de la variable (autre que "")
    for col in object_col_with_nan:
        # Récupère les différentes valeurs pour la série
        most_val = app_train_df[col].value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True).index
        # Si la première n'est pas vide
        if most_val[0]!="":
            # La prend pour remplir les valeurs vides
            most_val=most_val[0]
        else :
            # Sinon, prend la deuxième valeur pour remplir les valeurs vides
            most_val=most_val[1]
        # Remplace les valeurs vides
        app_train_df.loc[app_train_df[col]=="", col]=most_val

    print("\t\tEncodage des variables catégorielles")
    # Label Encoding pour toutes les variables catégorielles avec seulement 2 catégories => fonction [`LabelEncoder` de Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
    # One-Hot Encoding pour toutes les variables catégorielles avec plus de 2 catégories => fonction [`get_dummies(df)` de Pandas](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html).
    print("\nEffectue l'encodage via label encoder des variables catégorielles <=2 catégories")

    # Cree une instance de label encoder
    le       = LabelEncoder()
    le_count = 0

    # Itere sur les colonnes
    for col in app_train_df.columns:
        if app_train_df[col].dtype == 'object':
            # Si le nombre de catégories est <=2
            if len(list(app_train_df[col].unique())) <= 2:
                # Fait la transformation
                print(f"\tTransforme la colonne {col}")
                app_train_df[col] = le.fit_transform(app_train_df[col])
                le_count += 1

    print(f"\t{le_count} colonnes ont été transformées avec label encoder.")

    print("\nEffectue l'encodage \"One-Hot Encoder\" via get_dummies de Pandas des variables catégorielles >2 catégories")

    # one-hot encoding of categorical variables
    app_train_df = pd.get_dummies(app_train_df)

    print(f"\tLe fichier app_train_df contient désormais {app_train_df.shape[0]} lignes et {app_train_df.shape[1]} colonnes.")

    print("\nDomain Knowledge Features (Ajout de variables métiers)")
    app_train_df['DAYS_EMPLOYED_PERCENT'] = app_train_df['DAYS_EMPLOYED'] / app_train_df['DAYS_BIRTH']          # pourcentage de jours employés par rapport à l'âge du client
    app_train_df['INCOME_CREDIT_PERC']    = app_train_df['AMT_INCOME_TOTAL'] / app_train_df['AMT_CREDIT']       # pourcentage du montant du crédit par rapport au revenu d'un client
    app_train_df['INCOME_PER_PERSON']     = app_train_df['AMT_INCOME_TOTAL'] / app_train_df['CNT_FAM_MEMBERS']  # pourcentage du montant du crédit par individus composant le foyer du client
    app_train_df['ANNUITY_INCOME_PERC']   = app_train_df['AMT_ANNUITY'] / app_train_df['AMT_INCOME_TOTAL']      # pourcentage de la rente du prêt par rapport au revenu d'un client
    app_train_df['PAYMENT_RATE']          = app_train_df['AMT_ANNUITY'] / app_train_df['AMT_CREDIT']            # pourcentage du montant du crédit remboursé annuellement

    # Cree un dataframe pour les features polynomial
    poly_features = app_train_df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

    # Cree les features polynomial de degree 3
    poly_transformer = PolynomialFeatures(degree=3)

    # Train the polynomial features
    poly_transformer.fit(poly_features)

    # Transforme les features
    poly_features = poly_transformer.transform(poly_features)

    # Cree un dataframe des nouvelles features
    poly_features = pd.DataFrame(poly_features, columns=poly_transformer.get_feature_names_out(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']))

    # Merge les variables polynomial dans le dataframe général
    poly_features['SK_ID_CURR'] = app_train_df['SK_ID_CURR']
    app_train_df                = app_train_df.merge(poly_features, on = 'SK_ID_CURR', how = 'left')

    # Pour finir : remplace les dernières valeurs vide par 0
    app_train_df = app_train_df.fillna(0)

    # Change columns names ([LightGBM] Do not support special JSON characters in feature name.)
    new_names    = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in app_train_df.columns}
    new_n_list   = list(new_names.values())
    # [LightGBM] Feature appears more than one time.
    new_names    = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col for i, (col, new_col) in enumerate(new_names.items())}
    app_train_df = app_train_df.rename(columns=new_names)

    print("Renomme les colonnes qui contiennent des caractères speciaux JSON")

    # Affiche les nouvelles dimensions
    print(f"\nLe dataframe app_train_df final contient {app_train_df.shape[0]} lignes et {app_train_df.shape[1]} colonnes.")

    return app_train_df
