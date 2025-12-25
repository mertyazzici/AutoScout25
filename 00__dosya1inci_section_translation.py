import pandas as pd

df = pd.read_csv(r"Bitirme\_1.csv")

from deep_translator import GoogleTranslator
from tqdm import tqdm
tqdm.pandas()
def translate_long_text(text, max_chunk_size=4500):
    text = str(text) if pd.notna(text) else ""
    chunks = []

    for i in range(0, len(text), max_chunk_size):
        chunk = text[i:i + max_chunk_size]
        try:
            translation = GoogleTranslator(source='auto', target='en').translate(chunk)
            # Eğer None dönerse boş string kullan
            if translation is None:
                translation = ""
        except Exception as e:
            print(f"Translation error: {e}")
            translation = ""
        chunks.append(translation)

    return " ".join(chunks)

df["description_en"] = df["description"].progress_apply(translate_long_text)

df = df.drop(columns=["description"])

df.to_csv(r"Bitirme\_1_translated.csv", index=False)

df["model_version_en"] = df["model_version"].progress_apply(translate_long_text)
df.to_csv(r"Bitirme\_1_translated_model_version.csv", index=False)


######################################################################################################################
# Colour

import pandas as pd
df = pd.read_csv(r"Bitirme\_1_translated_model_version.csv")

from deep_translator import GoogleTranslator
from tqdm import tqdm
tqdm.pandas()
def translate_long_text(text, max_chunk_size=4500):
    text = str(text) if pd.notna(text) else ""
    chunks = []

    for i in range(0, len(text), max_chunk_size):
        chunk = text[i:i + max_chunk_size]
        try:
            translation = GoogleTranslator(source='auto', target='en').translate(chunk)
            # Eğer None dönerse boş string kullan
            if translation is None:
                translation = ""
        except Exception as e:
            print(f"Translation error: {e}")
            translation = ""
        chunks.append(translation)

    return " ".join(chunks)

df["body_color_original_en"] = df["body_color_original"].progress_apply(translate_long_text)
df.to_csv(r"Bitirme\_1_translated_model_version2.csv", index=False)