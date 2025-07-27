from transformers import pipeline
import os
import csv
import torch
import gc

# Verzeichnis mit den Audiodateien
data_dir = "eval_data_dir"
csv_dir = "csv_dir" # Pfad, an dem die CSV abgelegt werden soll

# Modell und Pipeline initialisieren
model_id = "erik1003/unispeech-sat-large-finetuned-birds-eu-10" # anpassen für anderes Model
pipe = pipeline("audio-classification", model=model_id, batch_size=8)

# Labels und CSV-Feldnamen vorbereiten
labels = list(pipe.model.config.id2label.values())  # Alle Speziesnamen aus dem Modell
score_fieldnames = [f"score_{l}" for l in labels]

counter = 0
correct_preds = 0

# Öffne die CSV-Datei im Schreibmodus
with open(csv_dir, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=["correct", "species", "file path", "prediction", "confidence"] + score_fieldnames)
    writer.writeheader()  # Schreibe die Kopfzeile

    # Durchlaufe alle Dateien im Verzeichnis rekursiv
    for root, _, files in os.walk(data_dir):
        for file_name in files:
            path = os.path.join(root, file_name)

            try:
                # Extrahiere den Speziesnamen aus dem Dateipfad
                species = os.path.basename(os.path.dirname(path))  # Ordnername als Speziesname

                with torch.no_grad():  # Verhindert unnötigen Speicherverbrauch
                    preds = pipe(path)
                counter += 1

                # Überprüfe, ob die Vorhersage korrekt ist
                correct = False
                if preds[0]["label"] == species:
                    correct_preds += 1
                    correct = True

                # Speichere die Scores in einem Dictionary
                score_dict = {f"score_{l}": 0.0 for l in labels}  # Default: 0.0
                for pred in preds:
                    score_dict[f"score_{pred['label']}"] = pred['score']

                # Schreibe die Ergebnisse in die CSV-Datei
                row = {
                    'correct': correct,
                    'species': species,
                    'file path': path,
                    'prediction': preds[0]["label"],
                    'confidence': preds[0]["score"],
                    **score_dict  # Alle score_xxx
                }
                writer.writerow(row)
                print(preds[0])

                # Speicher freigeben
                del preds
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"Fehler bei Datei {path}: {e}")

    print(f"Bisher wurden {correct_preds} von {counter} Dateien korrekt vorhergesagt")
    if counter != 0:
        print(f"Das entspricht einer Genauigkeit von {correct_preds / counter}")

print(f"Insgesamt wurden {correct_preds} von {counter} Dateien korrekt vorhergesagt")
print(f"Das entspricht einer Genauigkeit von {correct_preds / counter}")
