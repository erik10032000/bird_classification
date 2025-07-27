import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score, precision_score, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
import matplotlib.pyplot as plt
import numpy as np
import warnings

model_name = "unispeech-sat-large-eu" # anpassen für anderes Modell
csv_path = "path_to_csv"
confusion_matrix_top_n = 15

df = pd.read_csv(csv_path)

# Alle Klassen extrahieren (aus den score-Spaltennamen)
class_names = [col.replace('score_', '') for col in df.columns if col.startswith('score_')]

# Ground truth binarisieren (One-hot, jede Klasse gegen Rest)
y_true = label_binarize(df['species'], classes=class_names)
y_score = df[[f"score_{c}" for c in class_names]].values

# ROC und AUC pro Klasse berechnen und fehlende Klassen markieren
auc_per_class = []

for i, c in enumerate(class_names):
    if y_true[:, i].sum() == 0:
        auc_per_class.append(np.nan)  # Oder 0, falls du 0 setzen möchtest
        continue
    fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    auc_per_class.append(roc_auc)

# Makro-AUC berechnen (NaN ignorieren)
makro_auc = np.nanmean(auc_per_class)  # NaN ignorieren
print('Makro-AUC:', makro_auc)

# Mikro-ROC (alle Entscheidungen als Flat-Vector)
fpr, tpr, _ = roc_curve(y_true.ravel(), y_score.ravel())
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr)
print('Mikro-AUC:', roc_auc)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.0])
plt.xlabel('FP-Rate')
plt.ylabel('Sensitivität')
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.show()
plt.savefig(model_name + "-roc.svg")

# Gewichteten AUC-Wert berechnen
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
weighted_auc_ovr = roc_auc_score(y_true, y_score, average='weighted', multi_class='ovr')
print(f"Gewichtete AUC (OVR): {weighted_auc_ovr:.4f}")

### Konfusionsmatrix erstellen

# Klassen filtern, die in den Ground-Truth-Daten vorkommen
valid_classes = df['species'].unique()
filtered_class_names = [c for c in class_names if c in valid_classes]

# Diskrete Klassenlabels aus den Scores ableiten
y_pred = np.argmax(y_score, axis=1)  # Index der höchsten Wahrscheinlichkeit
y_true_labels = np.argmax(y_true, axis=1)  # Ground-Truth-Klassenlabels

# Die n größten Klassen basierend auf der Häufigkeit in den Ground-Truth-Daten auswählen
class_counts = df['species'].value_counts()
top_classes = class_counts.index[:confusion_matrix_top_n]  # Die n häufigsten Klassen

# Filtere die Ground-Truth- und Vorhersage-Labels basierend auf den Top-n-Klassen
class_to_index = {cls: idx for idx, cls in enumerate(class_names)}
selected_indices = [class_to_index[cls] for cls in top_classes if cls in class_to_index]

# Labels für die Top-n-Klassen extrahieren
y_true_filtered = []
y_pred_filtered = []

for true_label, pred_label in zip(y_true_labels, y_pred):
    if true_label in selected_indices and pred_label in selected_indices:
        y_true_filtered.append(true_label)
        y_pred_filtered.append(pred_label)

# Überprüfen, welche Klassen tatsächlich in den gefilterten Daten vorkommen
actual_classes = list(set(y_true_filtered) | set(y_pred_filtered))
actual_class_names = [class_names[idx] for idx in selected_indices if idx in actual_classes]

# Konfusionsmatrix anzeigen mit den Top-n-Klassen (nach Häufigkeit sortiert)
#plt.figure(figsize=(15, 13))  # Vergrößert den Plot (Breite x Höhe in Zoll)
disp = ConfusionMatrixDisplay.from_predictions(
    y_true_filtered, 
    y_pred_filtered, 
    display_labels=actual_class_names,  # Nach Häufigkeit sortierte Klassen
    cmap=plt.cm.Blues,
    colorbar=False,
)
disp.plot()
for labels in disp.text_.ravel():
    labels.set_fontsize(8)  # Schriftgröße der Labels in der Matrix anpassen

# plt.title(f'Konfusionsmatrix (Top-{confusion_matrix_top_n} Klassen)')
#print(f"Klassen in der Konfusionsmatrix: {actual_class_names}")

# Achsenbeschriftungen anpassen
plt.xticks(rotation=45, ha='right', fontsize=7)  # Dreht die x-Achsen-Beschriftungen
plt.yticks(fontsize=7)  # Passt die Schriftgröße der y-Achsen-Beschriftungen an
plt.xlabel('vorhergesagte Klasse')
plt.ylabel('tatsächliche Klasse')

# Layout anpassen, um abgeschnittene Labels zu vermeiden
plt.tight_layout()

# Plot speichern und anzeigen
plt.savefig(f"{model_name}-confusion-matrix-top-{confusion_matrix_top_n}.svg")
plt.show()

# Top-1 Accuracy berechnen
top1_accuracy = accuracy_score(y_true_labels, y_pred)
print(f"Top-1 Accuracy: {top1_accuracy:.4f}")

# F1-Score berechnen (Makro- und Mikro-Durchschnitt)
f1_macro = f1_score(y_true_labels, y_pred, average='macro')
f1_weighted = f1_score(y_true_labels, y_pred, average='weighted')
print(f"F1-Score (Makro): {f1_macro:.4f}")
print(f"F1-Score (Gewichtet): {f1_weighted:.4f}")

# Precision berechnen (Makro- und Mikro-Durchschnitt)
precision_macro = precision_score(y_true_labels, y_pred, average='macro', zero_division=0)
precision_weighted = precision_score(y_true_labels, y_pred, average='weighted', zero_division=0)
print(f"Precision (Makro): {precision_macro:.4f}")
print(f"Precision (Gewichtet): {precision_weighted:.4f}")

# Recall berechnen (Makro- und Mikro-Durchschnitt)
recall_macro = recall_score(y_true_labels, y_pred, average='macro', zero_division=0)
recall_weighted = recall_score(y_true_labels, y_pred, average='weighted', zero_division=0)
print(f"Recall (Makro): {recall_macro:.4f}")
print(f"Recall (Gewichtet): {recall_weighted:.4f}")
