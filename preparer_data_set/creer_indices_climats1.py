import ee
import pandas as pd
import matplotlib.pyplot as plt

# Initialiser Earth Engine
ee.Initialize(project='project37246')

# Définir la région du Sahel
# = ['Senegal', 'Mauritania', 'Mali', 'Burkina Faso', 'Niger', 'Chad', 'Sudan', 'Eritrea']
sahel_countries = ['Senegal']
sahel = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017") \
    .filter(ee.Filter.inList('country_na', sahel_countries))

years = list(range(2000, 2017))

# === Fonctions ===
def get_chirps_total(year):
    try:
        img = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
            .filterDate(f'{year}-06-01', f'{year}-11-30') \
            .filterBounds(sahel).sum().select('precip')
        val = img.reduceRegion(ee.Reducer.mean(), sahel.geometry(), 10000)
        return val.getInfo().get('precip')
    except:
        return None

def get_lst(year):
    try:
        img = ee.ImageCollection("MODIS/061/MOD11A2") \
            .filterDate(f'{year}-06-01', f'{year}-11-30') \
            .filterBounds(sahel).mean().select("LST_Day_1km")
        val = img.reduceRegion(ee.Reducer.mean(), sahel.geometry(), 1000)
        lst = val.getInfo().get('LST_Day_1km')
        return lst / 10 if lst else None
    except:
        return None

def get_ndvi(year):
    try:
        img = ee.ImageCollection("MODIS/061/MOD13A2") \
            .filterDate(f'{year}-06-01', f'{year}-11-30') \
            .filterBounds(sahel).mean().select("NDVI")
        val = img.reduceRegion(ee.Reducer.mean(), sahel.geometry(), 1000)
        ndvi = val.getInfo().get('NDVI')
        return ndvi / 10000 if ndvi else None
    except:
        return None

def get_soil_moisture(year):
    try:
        img = ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M") \
            .filterDate(f'{year}-06-01', f'{year}-11-30') \
            .filterBounds(sahel).mean().select("SoilMoi0_10cm_inst")
        val = img.reduceRegion(ee.Reducer.mean(), sahel.geometry(), 10000)
        return val.getInfo().get('SoilMoi0_10cm_inst')
    except:
        return None

def get_sos_doy(year):
    try:
        daily = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
            .filterDate(f'{year}-06-01', f'{year}-11-30') \
            .filterBounds(sahel).sort("system:time_start")

        def add_doy(img):
            doy = ee.Date(img.get("system:time_start")).getRelative("day", "year")
            return img.set("doy", doy)

        daily = daily.map(add_doy)
        rainy_days = daily.map(lambda img: img.updateMask(img.select("precip").gte(3)))
        doy_list = rainy_days.aggregate_array("doy").getInfo()

        doy_list = sorted(set(int(d) for d in doy_list)) if doy_list else []
        for i in range(len(doy_list) - 2):
            if doy_list[i + 1] == doy_list[i] + 1 and doy_list[i + 2] == doy_list[i] + 2:
                return doy_list[i]
        return None
    except:
        return None

def get_fldas_indices(year):
    try:
        bands = [
            "Evap_tavg", "Qair_f_tavg", "SoilTemp_tavg", "Rainf_f_tavg",
            "SoilMoi0_10cm_inst", "SoilMoi10_40cm_inst", "SoilMoi40_100cm_inst",
            "SoilMoi100_200cm_inst", "Qg_tavg", "SWdown_f_tavg", "LWdown_f_tavg"
        ]
        img = ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M") \
            .filterDate(f'{year}-06-01', f'{year}-11-30') \
            .filterBounds(sahel).mean().select(bands)
        val = img.reduceRegion(ee.Reducer.mean(), sahel.geometry(), 10000)
        return val.getInfo()
    except:
        return {}

# === Collecte des données ===
data = []
for year in years:
    print(f"🔄 Traitement de l’année {year}...")
    record = {'year': year}
    record['precip_mm'] = get_chirps_total(year)
    record['lst_celsius'] = get_lst(year)
    record['ndvi'] = get_ndvi(year)
    record['soil_moisture'] = get_soil_moisture(year)
    record['sos_doy'] = get_sos_doy(year)
    fldas = get_fldas_indices(year)
    for k, v in fldas.items():
        record[k] = v
    data.append(record)

# === Construction du DataFrame ===
df = pd.DataFrame(data)

# Ajout des colonnes FLDAS manquantes (sécurité)
fldas_cols = [
    "Evap_tavg", "Qair_f_tavg", "SoilTemp_tavg", "Rainf_f_tavg",
    "SoilMoi0_10cm_inst", "SoilMoi10_40cm_inst", "SoilMoi40_100cm_inst",
    "SoilMoi100_200cm_inst", "Qg_tavg", "SWdown_f_tavg", "LWdown_f_tavg"
]
for col in fldas_cols:
    if col not in df.columns:
        df[col] = None

# Fusion avec ONI
oni_annual = pd.read_csv("oni_annual_1981_2020.csv")
df_final = df.merge(oni_annual, on="year", how="left")

# === Sauvegarde Excel des données ===
df_final.to_excel("saison_sahel_2000_2016_12.xlsx", index=False)
print("✅ Données enregistrées.")

# === Visualisation conditionnelle ===
figures = []

def plot_and_store(x, y, xlabel, ylabel, title):
    if x not in df_final.columns or y not in df_final.columns:
        print(f"⚠️ Impossible de tracer {title} : colonne(s) manquante(s) ({x}, {y})")
        return
    if df_final[[x, y]].dropna().empty:
        print(f"⚠️ Données insuffisantes pour tracer {title}")
        return
    fig, ax = plt.subplots()
    ax.plot(df_final[x], df_final[y], marker='o', linestyle='-')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    figures.append((title, fig))

# Tracés robustes
plot_and_store('year', 'ndvi', 'Année', 'NDVI', 'NDVI moyen par an')
plot_and_store('year', 'precip_mm', 'Année', 'Précipitations (mm)', 'Précipitations annuelles')
plot_and_store('ndvi', 'precip_mm', 'NDVI', 'Précipitations', 'NDVI vs Précipitations')
plot_and_store('lst_celsius', 'ndvi', 'LST (°C)', 'NDVI', 'Température de surface vs NDVI')
plot_and_store('oni', 'precip_mm', 'ONI', 'Précipitations', 'ONI vs Précipitations')
plot_and_store('lst_celsius', 'Evap_tavg', 'LST (°C)', 'Evapotranspiration (mm)', 'LST vs Evapotranspiration')

# === Export Excel avec graphiques ===
with pd.ExcelWriter('saison_sahel_2000_2016_graphs.xlsx', engine='xlsxwriter') as writer:
    df_final.to_excel(writer, index=False, sheet_name='Indices_Sahel')
    workbook = writer.book
    for title, fig in figures:
        sheet_name = title[:30]
        worksheet = workbook.add_worksheet(sheet_name)
        writer.sheets[sheet_name] = worksheet
        img_path = f"{sheet_name}.png"
        fig.savefig(img_path, bbox_inches='tight')
        worksheet.insert_image('B2', img_path)
        plt.close(fig)

print("✅ Fichier Excel avec graphiques généré.")
