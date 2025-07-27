from transformers import pipeline
import urllib.request
import json
import gradio as gr
import wikipedia
from duckduckgo_search import DDGS
from fastcore.all import *

model_id = "erik1003/unispeech-sat-large-finetuned-birds-eu-10" # anpassen für anderes Model
pipe = pipeline("audio-classification", model=model_id)
wikipedia.set_lang("de")

def get_locale_data(locale: str):
    """Download eBird locale species data.
    Requests the locale data through the eBird API.
    Args:
        locale: Two character string of a language.
    Returns:
        A data object containing the response from eBird.
    """
    url = "https://api.ebird.org/v2/ref/taxonomy/ebird?cat=species&fmt=json&locale=" + locale

    req = urllib.request.Request(url)
    response = urllib.request.urlopen(req)

    return json.loads(response.read())

bird_dict = dict()
de = get_locale_data("de")
print("Deutsche Label fertig heruntergeladen")
en = get_locale_data("en")
print("Englische Label fertig heruntergeladen")

for bird in en:
    for vogel in de:
        if bird["sciName"] == vogel["sciName"]:    
            bird_dict[bird["comName"]] = vogel["comName"]
            break

print("Dict fertig erstellt")


def get_wikipedia_summary(bird_name):
    """Fetches a short Wikipedia summary for the given bird name."""
    try:
        summary = wikipedia.summary(bird_name, sentences=2, auto_suggest=True)
        return summary
    except:
        return "Keine Wikipedia-Informationen gefunden."

def search_images(keywords, max_images=5): return L(DDGS().images(keywords, max_results=max_images)).itemgot('image')

def classify_audio(filepath):
    preds = pipe(filepath)
    outputs = {}
    for p in preds[:5]:
        try:
            outputs[bird_dict[p["label"]]] = p["score"]
        except KeyError:
            print("KeyError bei " + p["label"])
            outputs[p["label"]] = p["score"]
    print("Predictions fertig")
    wiki_info = get_wikipedia_summary(bird_dict[preds[0]["label"]])
    print("Wikipedia fertig")
    try:
        image_urls = search_images(preds[0]["label"])
        print("Bilder fertig")
    except Exception as e:
        print(e)
        image_urls = ["https://upload.wikimedia.org/wikipedia/commons/a/a3/Image-not-found.png?20210521171500"]
    
    
    return image_urls, wiki_info, outputs


demo = gr.Interface(
    fn=classify_audio,
    inputs=gr.Audio(type="filepath"),
    outputs=[gr.Gallery(label="DuckDuckGo Bilder"), gr.Textbox(label="Wikipedia Info"), gr.Label()]
)
demo.launch()
