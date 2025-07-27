from datasets import load_dataset
from multiprocessing import cpu_count
from transformers import AutoFeatureExtractor
from datasets import Audio
from transformers import AutoModelForAudioClassification
from huggingface_hub import login
from transformers import TrainingArguments
import evaluate
import numpy as np
from transformers import Trainer

batch_size = 48
gradient_accumulation_steps = 4
num_train_epochs = 10
model_id = "hugging_face_model_path"
data_dir = "data_dir_path"

login(token="TOKEN")

# Datensatz laden
dataset = load_dataset("audiofolder", data_dir=data_dir)
dataset = dataset["train"].train_test_split(test_size=0.2)

# jedem Label eine ID zuweisen
id2label = dataset["train"].features["label"].int2str
id2label(dataset["train"][0]["label"])

# Checkpoint von Huggingface laden
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id, do_normalize=True,
)

# Audiodateien vorbearbeiten
sampling_rate = feature_extractor.sampling_rate # Samplingrate des vortrainierten Models
dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate)) # Alle Dateien werden on-the-fly in die passende Sampling rate umgewandelt

max_duration = 30.0
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        truncation=True,
    )
    return inputs 

dataset_encoded = dataset.map(
    preprocess_function,
    remove_columns=["audio"],
    batched=True,
    batch_size=16,
)

id2label = {
    str(i): id2label(i)
    for i in range(len(dataset_encoded["train"].features["label"].names))
}
label2id = {v: k for k, v in id2label.items()}

print(str(len(id2label)) + " Label gelesen")
num_labels = len(id2label)

# Model initialisieren
model = AutoModelForAudioClassification.from_pretrained(
    model_id,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
)

model_name = model_id.split("/")[-1]

training_args = TrainingArguments(
    run_name = f"{model_name}-finetuned-birds-de-10",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    warmup_ratio=0.1,
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    bf16=True,  # H/A100 unterstützt BF16 effizient statt fp16=True,
    push_to_hub=True, #true
    output_dir="share/working/" + f"{model_name}-finetuned-birds-de-10",
    save_total_limit=3,
    dataloader_num_workers=1,
    logging_strategy="steps",
)

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

# Training starten
trainer = Trainer(
    model,
    training_args,
    train_dataset=dataset_encoded["train"],
    eval_dataset=dataset_encoded["test"],
    processing_class=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()
