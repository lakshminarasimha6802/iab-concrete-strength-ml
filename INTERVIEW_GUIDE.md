# Explain your project to IAB: interview notes

## 60-second German pitch (use only AFTER running the real project)

„Ich habe ein kleines Machine-Learning-Projekt zur Vorhersage der Betondruckfestigkeit entwickelt. Die Daten stammen aus dem öffentlichen UCI-Datensatz. Mit Python und pandas habe ich die Daten geladen, geprüft und analysiert. Anschließend habe ich lineare Regression und Random Forest mit Scikit-learn verglichen. Die Auswahl erfolgte mit einer fünffachen Kreuzvalidierung auf Trainingsdaten. Am Ende habe ich das beste Modell auf separaten Testdaten bewertet. Die Ergebnisse und Einschränkungen habe ich dokumentiert.“

## Key questions

**Warum Regression?** Die Zielvariable ist eine kontinuierliche Zahl in MPa. Klassifikation sagt Kategorien voraus, Regression einen numerischen Wert.

**Was ist pandas?** Eine Python-Bibliothek für tabellarische Daten. Ich nutze DataFrames, CSV-Einlesen, Qualitätsprüfungen und Datenexport.

**Was ist Labeling hier?** Zusätzliche relative Gruppen (lower/middle/upper) nur für Exploration. Sie leiten sich aus der Zielvariable ab und dürfen deshalb nicht für die Vorhersage als Input benutzt werden. Es sind keine normierten Festigkeitsklassen.

**Wieso Daten vor dem Training prüfen?** Fehlerhafte Einheiten, fehlende Spalten, ungültige Zahlen und Ausreißer können die Ergebnisse verfälschen. Duplikate sehe ich mir an und entferne sie nicht automatisch.

**Was ist Data Leakage?** Informationen aus Testdaten oder aus der Zielvariable gelangen unerlaubt in den Modellaufbau. Deshalb wird erst getrennt und die Vorverarbeitung nur an Trainingsdaten angepasst.

**Warum zwei Modelle?** Linear Regression ist eine einfache, interpretierbare Basislinie; Random Forest kann nichtlineare Zusammenhänge abbilden.

**Was ist Kreuzvalidierung?** Das Trainingsset wird in fünf Teile aufgeteilt; ein Teil dient zur Validierung, vier zum Trainieren. Das wird fünfmal wiederholt, um stabilere Auswahlkriterien zu erhalten.

**Was bedeuten MAE, RMSE und R²?** MAE ist die mittlere absolute Abweichung in MPa; RMSE bestraft große Fehler stärker; R² misst die Verbesserung gegenüber einer konstanten Mittelwert-Vorhersage (kann negativ sein).

**Kann man das Modell praktisch am Bau verwenden?** Nicht ohne unabhängige Validierung. Der Datensatz ist historisch und klein, und Normen/Materialeigenschaften können abweichen.

**Was fehlt dem Projekt?** Eine echte externe Validierung, Rezeptur-Gruppierung, Unsicherheitsanalyse sowie weitere Datenquellen. Keine reale Sensor-Datenakquise oder manuelles Bildlabeling.

## Demonstration checklist

1. Open README; explain UCI attribution and eight inputs.
2. Run `python -m pytest -q`.
3. Run `python -m src.train_model` with cached dataset.
4. Open `reports/results.md`; read your ACTUAL MAE/RMSE/R².
5. Open `actual_vs_predicted.png`, explain deviation from dashed diagonal.
6. Discuss what would make the model less reliable on new construction mixes.
