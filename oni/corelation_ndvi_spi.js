// Script pour analyser la corrélation entre NDVI et SPI au Sénégal
// Ce script utilise les collections d'images MODIS pour NDVI et CHIRPS pour SPI
//usage: Google Earth Engine Code Editor
// Définir la région du Sénégal
var senegal = ee.FeatureCollection('FAO/GAUL_SIMPLIFIED_500m/2015/level1')
  .filter(ee.Filter.eq('ADM0_NAME', 'Senegal'));

// Charger NDVI MODIS (2018-2020)
var ndviCollection = ee.ImageCollection('MODIS/006/MOD13Q1')
  .filterDate('2018-01-01', '2020-12-31')
  .select('NDVI')
  .map(image => image
    .multiply(0.0001)
    .rename('NDVI')
    .clip(senegal)
    .copyProperties(image, ['system:time_start'])
  );

// Charger précipitations CHIRPS
var spiCollection = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterDate('2018-01-01', '2020-12-31')
  .select('precipitation')
  .map(image => image
    .rename('SPI')
    .clip(senegal)
    .copyProperties(image, ['system:time_start'])
  );

// Reprojection pour homogénéiser la résolution
ndviCollection = ndviCollection.map(img => img.reproject({crs: 'EPSG:4326', scale: 1000}));
spiCollection = spiCollection.map(img => img.reproject({crs: 'EPSG:4326', scale: 1000}));

// Joindre les collections par date
var join = ee.Join.inner();
var filter = ee.Filter.equals({
  leftField: 'system:time_start',
  rightField: 'system:time_start'
});
var joined = join.apply(ndviCollection, spiCollection, filter);

// Fusionner les images jointes
var merged = ee.ImageCollection(joined.map(pair => {
  var ndvi = ee.Image(ee.Feature(pair).get('primary'));
  var spi = ee.Image(ee.Feature(pair).get('secondary'));
  return ndvi.addBands(spi).set('system:time_start', ndvi.get('system:time_start'));
}));

// Échantillonner 500 pixels dans la première image
var samplePoints = merged.first().sample({
  region: senegal,
  scale: 1000,
  numPixels: 500,
  geometries: true
});

// Afficher graphique corrélation
var chart = ui.Chart.feature.byFeature(samplePoints, 'SPI', ['NDVI'])
  .setChartType('ScatterChart')
  .setOptions({
    title: 'Corrélation NDVI vs SPI - Sénégal',
    hAxis: {title: 'SPI (Précipitation)'},
    vAxis: {title: 'NDVI'},
    pointSize: 4,
    trendlines: {0: {type: 'linear', color: 'red'}}
  });
print(chart);
