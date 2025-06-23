import ee
import pandas as pd

# Initialiser Earth Engine avec ton projet
ee.Initialize(project='project37246')

# Définir la région du Sahel
sahel_countries = [
    'Senegal', 'Mauritania', 'Mali', 'Burkina Faso', 'Niger',
    'Chad', 'Sudan', 'Eritrea'
]
sahel = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017") \
    .filter(ee.Filter.inList('country_na', sahel_countries))

years = list(range(2000, 2017))

# Fonctions d'extraction des indices
def get_chirps_total(year):
    start = ee.Date.fromYMD(year, 6, 1)
    end = ee.Date.fromYMD(year, 11, 30)
    img = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .filterDate(start, end) \
        .filterBounds(sahel) \
        .sum().select('precip')
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
    end = ee.Date.fromYMD(year, 11, 30)
    img = ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M") \
        .filterDate(start, end) \
        .filterBounds(sahel) \
        .mean().select("SoilMoi0_10cm_inst")
    val = img.reduceRegion(ee.Reducer.mean(), sahel.geometry(), 10000)
    return val.getInfo().get('SoilMoi0_10cm_inst')
def get_fldas_indices(year):
    start = ee.Date.fromYMD(year, 6, 1)
    end = ee.Date.fromYMD(year, 11, 30)
    bands = [
        "Evap_tavg", "Qair_f_tavg", "SoilTemp_tavg", "Rainf_f_tavg",
        "SoilMoi0_10cm_inst", "SoilMoi10_40cm_inst", "SoilMoi40_100cm_inst",
        "SoilMoi100_200cm_inst", "Qg_tavg", "SWdown_f_tavg", "LWdown_f_tavg"
    ]

    try:
        img = ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M/V001") \
            .filterDate(start, end) \
            .filterBounds(sahel) \
            .mean()

        # Vérifier si les bandes existent
        available_bands = img.bandNames().getInfo()
        missing_bands = [b for b in bands if b not in available_bands]

        if missing_bands:
            print(f"⚠️ {year} - Bandes manquantes : {missing_bands}")

        selected_img = img.select([b for b in bands if b in available_bands])

        val = selected_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=sahel.geometry(),
            scale=25000,  # Plus large que 10000
            maxPixels=1e13
        )
        result = val.getInfo()

        if not result:
            print(f"⚠️ {year} - Résultat FLDAS vide.")
            return {}

        return result

    except Exception as e:
        print(f"❌ {year} - Erreur FLDAS : {e}")
        return {}


def get_sos_doy(year):
    start = ee.Date.fromYMD(year, 6, 1)
    end = ee.Date.fromYMD(year, 11, 30)
    daily = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .filterDate(start, end) \
        .filterBounds(sahel) \
        .sort("system:time_start")

    def add_doy(img):
        doy = ee.Date(img.get("system:time_start")).getRelative("day", "year")
        return img.set("doy", doy)

    daily = daily.map(add_doy)
    rainy_days = daily.map(lambda img: img.updateMask(img.select("precip").gte(3)))
    doy_list = rainy_days.aggregate_array("doy").getInfo()

    if not doy_list or len(doy_list) < 2:
        return None

    doy_list = sorted(set(int(d) for d in doy_list))

    for i in range(len(doy_list) - 2):
        if doy_list[i + 1] == doy_list[i] + 1 and doy_list[i + 2] == doy_list[i] + 2:
            return doy_list[i]
    return None

# Collecte des données
data = []
for year in years:
    print(f"🔄 Traitement de l’année {year}...")
    record = {'year': year}

    try: record['precip_mm'] = get_chirps_total(year)
    except: record['precip_mm'] = None

    try: record['lst_celsius'] = get_lst(year)
    except: record['lst_celsius'] = None

    try: record['ndvi'] = get_ndvi(year)
    except: record['ndvi'] = None

    try: record['soil_moisture'] = get_soil_moisture(year)
    except: record['soil_moisture'] = None

    try: record['sos_doy'] = get_sos_doy(year)
    except: record['sos_doy'] = None

    try:
        fldas = get_fldas_indices(year)
        for k, v in fldas.items():
            record[k] = v
    except:
        pass

    data.append(record)

# Création du DataFrame
df = pd.DataFrame(data)

# Lecture et fusion avec les données ONI
oni_annual = pd.read_csv("oni_annual_1981_2020.csv")  # Assure-toi que ce fichier existe
df_final = df.merge(oni_annual, on="year", how="left")

# Export en format Excel (.xlsx)
output_file = "saison_sahel_2000_2016_1.xlsx"
df_final.to_excel(output_file, index=False)
print(f"✅ Données enregistrées dans : {output_file}")

# Aperçu
print(df_final.head())
import matplotlib.pyplot as plt

# 📊 Générer des graphiques
figures = []

def plot_and_store(x, y, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    ax.plot(df_final[x], df_final[y], marker='o', linestyle='-')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    figures.append((title, fig))

plot_and_store('year', 'ndvi', 'Année', 'NDVI', 'NDVI moyen par an')
plot_and_store('year', 'precip_mm', 'Année', 'Précipitations (mm)', 'Précipitations annuelles')
plot_and_store('ndvi', 'precip_mm', 'NDVI', 'Précipitations', 'NDVI vs Précipitations')
plot_and_store('lst_celsius', 'ndvi', 'LST (°C)', 'NDVI', 'Température de surface vs NDVI')
plot_and_store('oni', 'precip_mm', 'ONI', 'Précipitations', 'ONI vs Précipitations')
plot_and_store('lst_celsius', 'Evap_tavg', 'LST (°C)', 'Evapotranspiration (mm)', 'LST vs Evapotranspiration')

# 📝 Export vers Excel avec graphiques
excel_file = 'saison_sahel_2000_2016.xlsx'
with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
    df_final.to_excel(writer, index=False, sheet_name='Indices_Sahel')
    workbook  = writer.book

    # Ajouter chaque figure à une nouvelle feuille
    for title, fig in figures:
        sheet_name = title[:30]  # Limiter à 31 caractères pour Excel
        worksheet = workbook.add_worksheet(sheet_name)
        writer.sheets[sheet_name] = worksheet

        # Sauvegarde temporaire du graphique
        img_path = f"{sheet_name}.png"
        fig.savefig(img_path, bbox_inches='tight')
        worksheet.insert_image('B2', img_path)
        plt.close(fig)

print(f"✅ Fichier Excel avec graphiques généré : {excel_file}")
