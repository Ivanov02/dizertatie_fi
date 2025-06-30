import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import warnings
import re
import os
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

def load_model_and_tokenizer(model_id):
    print(f"Se încarcă modelul: {model_id}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Modelul și tokenizer-ul au fost încărcate cu succes.")
    print(f"Modelul este pe: {model.device}")
    return model, tokenizer

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_zero_shot_inference(text_generator, article, target):
    prompt = (
        f"Task: Identify the stance of the following text towards the target.\n"
        f"Return only one of: Pro, Against, Neutral.\n\n"
        f"Text: \"{article}\"\n"
        f"Target: \"{target}\"\n\n"
        f"Answer:"
    )
    outputs = text_generator(
        prompt,
        max_new_tokens=10,
        do_sample=False,
        truncation=True,
        max_length=512
    )
    raw_prediction = outputs[0]['generated_text'].strip()
    match = re.search(r"\b(pro|against|neutral)\b", raw_prediction, re.IGNORECASE)
    return match.group(1).capitalize() if match else None

def main():
    model_id = "google/flan-t5-large"
    model, tokenizer = load_model_and_tokenizer(model_id)

    text_generator = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer
    )

    test_data = load_data("../data_sources/ro_stance_test_2.json")

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "flan_t5_large_float16_auto_predictions_full_file_eng_ro.json")

    mapping_eng_to_ro = {
        "Pro": "Pro",
        "Against": "Contra",
        "Neutral": "Neutru"
    }

    results_to_save = []
    correct_predictions = 0
    total_predictions = 0
    invalid_predictions = 0

    total_annotations = sum(len(entry.get("annotations", [])) for entry in test_data)

    print(f"\nSe începe inferența cu modelul {model_id} pentru întregul fișier...")
    with tqdm(total=total_annotations, desc="Inferență articole", ncols=80) as pbar:
        for entry in test_data:
            article_text = entry.get("article", "Text lipsă")
            annotations = entry.get("annotations", [])

            for annotation in annotations:
                target = annotation.get("target")
                true_label = annotation.get("label")

                # Sari peste adnotările incomplete
                if not target or not true_label:
                    continue

                predicted_label = run_zero_shot_inference(text_generator, article_text, target)
                total_predictions += 1

                is_valid = predicted_label is not None
                mapped_prediction = mapping_eng_to_ro.get(predicted_label, "None")
                is_correct = (mapped_prediction == true_label) if is_valid else False

                if not is_valid:
                    invalid_predictions += 1

                if is_correct:
                    correct_predictions += 1

                results_to_save.append({
                    "article": article_text,
                    "target": target,
                    "true_label": true_label,
                    "predicted_label": mapped_prediction,
                    "is_correct": is_correct
                })
                pbar.update(1)

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"\n--- Rezumat Final ---")
    print(f"Total predicții: {total_predictions}")
    print(f"Predicții corecte: {correct_predictions}")
    print(f"Predicții invalide: {invalid_predictions}")
    print(f"Acuratețe totală: {accuracy:.2f}%")

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=4, ensure_ascii=False)
    print(f"Rezultatele au fost salvate în: {output_file_path}")

if __name__ == "__main__":
    main()