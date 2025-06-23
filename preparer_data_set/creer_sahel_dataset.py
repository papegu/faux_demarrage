import ee
import pandas as pd

# Initialiser Earth Engine
ee.Initialize(project='project37246')

# Définir les pays du Sahel
sahel_countries = [
    'Senegal', 'Mauritania', 'Mali', 'Burkina Faso', 'Niger',
    'Chad', 'Sudan', 'Eritrea'
]

sahel = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017") \
    .filter(ee.Filter.inList('country_na', sahel_countries))

years = list(range(2000, 2017))

# Fonctions d'extraction mises à jour
def get_chirps_total(year):
    start = ee.Date.fromYMD(year, 6, 1)
    end = ee.Date.fromYMD(year, 11, 30)
    img = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .filterDate(start, end) \
        .filterBounds(sahel) \
        .sum().select('precip')  # Ajout .select('precip') ici
    val = img.reduceRegion(ee.Reducer.mean(), sahel.geometry(), 10000)
    try:
        return val.getInfo().get('precip')
    except:
        return None

def get_lst(year):
    start = ee.Date.fromYMD(year, 6, 1)
    end = ee.Date.fromYMD(year, 11, 30)
    img = ee.ImageCollection("MODIS/061/MOD11A2") \
        .filterDate(start, end) \
        .filterBounds(sahel) \
        .mean().select("LST_Day_1km")
    val = img.reduceRegion(ee.Reducer.mean(), sahel.geometry(), 1000)
    lst = val.getInfo().get('LST_Day_1km')
    return lst / 10 if lst is not None else None

def get_ndvi(year):
    start = ee.Date.fromYMD(year, 6, 1)
    end = ee.Date.fromYMD(year, 11, 30)
    img = ee.ImageCollection("MODIS/061/MOD13A2") \
        .filterDate(start, end) \
        .filterBounds(sahel) \
        .mean().select("NDVI")
    val = img.reduceRegion(ee.Reducer.mean(), sahel.geometry(), 1000)
    ndvi = val.getInfo().get('NDVI')
    return ndvi / 10000 if ndvi is not None else None

def get_soil_moisture(year):
    start = ee.Date.fromYMD(year, 6, 1)
    end = ee.Date.fromYMD(year, 9, 30)
    img = ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M") \
        .filterDate(start, end) \
        .filterBounds(sahel) \
        .mean().select("SoilMoi0_10cm_inst")
    val = img.reduceRegion(ee.Reducer.mean(), sahel.geometry(), 10000)
    return val.getInfo().get('SoilMoi0_10cm_inst')
def get_fldas_indices(year):
    # Définir la période de la saison humide
    start = ee.Date.fromYMD(year, 6, 1)
    end = ee.Date.fromYMD(year, 11, 30)
    
    # Variables à extraire
    bands = [
        "Evap_tavg",             # Evapotranspiration
        "Qair_f_tavg",           # Humidité spécifique
        "SoilTemp_tavg",         # Température du sol
        "Rainf_f_tavg",          # Précipitations
        "SoilMoi0_10cm_inst",    # Humidité du sol 0-10 cm
        "SoilMoi10_40cm_inst",   # Humidité du sol 10-40 cm
        "SoilMoi40_100cm_inst",  # Humidité du sol 40-100 cm
        "SoilMoi100_200cm_inst", # Humidité du sol 100-200 cm
        "Qg_tavg",               # Flux de chaleur du sol
        "SWdown_f_tavg",         # Rayonnement solaire incident
        "LWdown_f_tavg"          # Rayonnement infrarouge incident
    ]
    
    # Charger et filtrer la collection
    img = ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M") \
        .filterDate(start, end) \
        .filterBounds(sahel) \
        .mean() \
        .select(bands)
    
    # Réduction spatiale pour obtenir la moyenne sur la région du Sahel
    val = img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=sahel.geometry(),
        scale=10000,
        maxPixels=1e13
    )
    
    return val.getInfo()

def get_sos_doy(year):
    start = ee.Date.fromYMD(year, 6, 1)
    end = ee.Date.fromYMD(year, 11,31)

    daily = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .filterDate(start, end) \
        .filterBounds(sahel) \
        .sort("system:time_start")

    # Ajouter le jour de l’année (DOY) à chaque image
    def add_doy(img):
        doy = ee.Date(img.get("system:time_start")).getRelative("day", "year")
        return img.set("doy", doy)

    daily = daily.map(add_doy)

    # Filtrer les jours avec précip ≥ 10 mm
    rainy_days = daily.map(lambda img: img.updateMask(img.select("precip").gte(3)))

    # Obtenir les DOY valides
    doy_list = rainy_days.aggregate_array("doy").getInfo()

    if not doy_list or len(doy_list) < 2:
        return None

    doy_list = sorted(set(int(d) for d in doy_list))

    # Chercher 3 jours consécutifs
    for i in range(len(doy_list) - 2):
        if doy_list[i + 1] == doy_list[i] + 1 and doy_list[i + 2] == doy_list[i] + 2:
            return doy_list[i]

    return None

# Collecte des données
data = []
for year in years:
    print(f"🔄 Traitement de l’année {year}...")
    try:
        precip = get_chirps_total(year)
    except:
        precip = None
    try:
        lst = get_lst(year)
    except:
        lst = None
    try:
        ndvi = get_ndvi(year)
    except:
        ndvi = None
    try:
        sm = get_soil_moisture(year)
    except:
        sm = None
    try:
        sos = get_sos_doy(year)
    except:
        sos = None

    data.append({
        'year': year,
        'precip_mm': precip,
        'lst_celsius': lst,
        'ndvi': ndvi,
        'soil_moisture': sm,
        'sos_doy': sos
    })

df = pd.DataFrame(data)

# Lecture des données ONI annuelles
oni_annual = pd.read_csv("oni_annual_1981_2020.csv")
df_final = df.merge(oni_annual, on="year", how="left")

# Sauvegarde
df_final.to_csv("saison_sahel_2000_2016_juin_octobre.csv", index=False)
print("✅ Fichier enregistré : 'saison_sahel_2000_2016.csv'")
print(df_final.head())
print("✅ Données du Sahel collectées et fusionnées avec ONI.")
# Affichage des premières lignes du DataFrame final
print(df_final.head())
# Affichage des colonnes du DataFrame final
print("Colonnes du DataFrame final :", df_final.columns.tolist())
# Affichage des informations du DataFrame final
print("Informations du DataFrame final :")
print(df_final.info())
# Affichage des statistiques descriptives du DataFrame final
print("Statistiques descriptives du DataFrame final :")
print(df_final.describe())
# Affichage des types de données du DataFrame final
print("Types de données du DataFrame final :")
print(df_final.dtypes)
# Affichage des valeurs manquantes du DataFrame final
print("Valeurs manquantes dans le DataFrame final :")
print(df_final.isnull().sum())      