import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import warnings
import re
import os
from tqdm import tqdm
import random

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

MODEL_CACHE_DIR = None


def load_model_and_tokenizer(model_id: str):

    print(f"Se încarcă modelul: {model_id}...")
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=MODEL_CACHE_DIR
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=MODEL_CACHE_DIR)
        # Asigură-te că tokenizer-ul are un pad_token, necesar pentru anumite operații.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Modelul și tokenizer-ul au fost încărcate cu succes.")
        print(f"Modelul este pe: {model.device}")
        return model, tokenizer
    except Exception as e:
        print(f"Eroare la încărcarea modelului sau a tokenizer-ului: {e}")
        return None, None


def load_data(file_path: str):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Eroare: Fișierul '{file_path}' nu a fost găsit. Asigură-te că calea este corectă.")
        return None
    except json.JSONDecodeError:
        print(f"Eroare: Nu s-a putut decoda fișierul JSON '{file_path}'. Verifică formatul fișierului.")
        return None
    except Exception as e:
        print(f"A apărut o eroare neașteptată la încărcarea datelor din '{file_path}': {e}")
        return None


def prepare_few_shot_examples(
        train_data: list,
        tokenizer: AutoTokenizer,
        num_examples_per_category: int = 1,  # <-- MODIFICAT AICI LA 1 EXEMPLU PER CATEGORIE
        max_example_text_length: int = 50
) -> str:

    examples_by_category = {"Pro": [], "Against": [], "Neutral": []}

    label_mapping_for_examples = {
        "Pro": "Pro",
        "Contra": "Against",
        "Neutru": "Neutral",
    }

    for entry in train_data:
        article_text = entry.get("article", "")
        annotations = entry.get("annotations", [])
        for annotation in annotations:
            target = annotation.get("target")
            label = annotation.get("label")

            formatted_label = label_mapping_for_examples.get(label)

            if target and formatted_label in ["Pro", "Against", "Neutral"]:
                truncated_text = tokenizer.decode(
                    tokenizer.encode(article_text, max_length=max_example_text_length, truncation=True),
                    skip_special_tokens=True
                )
                examples_by_category[formatted_label].append({
                    "text": truncated_text,
                    "target": target,
                    "label": formatted_label
                })

    few_shot_prompt_parts = []
    for category in ["Pro", "Against", "Neutral"]:
        examples = examples_by_category.get(category, [])
        random.shuffle(examples)
        for example in examples[:num_examples_per_category]:
            few_shot_prompt_parts.append(
                f"Text: \"{example['text']}\"\n"
                f"Target: \"{example['target']}\"\n"
                f"Answer: {example['label']}"
            )

    return "\n\n".join(few_shot_prompt_parts)


def run_few_shot_inference(
        text_generator: pipeline,
        article: str,
        target: str,
        few_shot_examples_str: str,
        debug: bool = False,
        tokenizer: AutoTokenizer = None
) -> str:

    prompt = (
        f"Task: Identify the stance of the following text towards the target.\n"
        f"Return only one of: Pro, Against, Neutral.\n\n"
        f"Examples:\n{few_shot_examples_str}\n\n"
        f"Text: \"{article}\"\n"
        f"Target: \"{target}\"\n\n"
        f"Answer:"
    )

    if tokenizer:
        prompt_length = len(tokenizer.encode(prompt, truncation=False))
        if prompt_length > 512:
            print(f"Lungimea promptului este {prompt_length} tokeni (> 512). "
                  f"Modelul va trunchia inputul. Reduceți max_example_text_length sau num_examples_per_category.")

    if debug:
        print("\n--- PROMPT CURENT ---")
        print(prompt)
        print("--------------------")

    outputs = text_generator(
        prompt,
        max_new_tokens=10,
        do_sample=False,
        truncation=True,
        max_length=512
    )
    raw_prediction = outputs[0]['generated_text'].strip()

    if debug:
        print(f"Predicție brută model: \"{raw_prediction}\"")

    match = re.search(r"\b(pro|against|neutral)\b", raw_prediction, re.IGNORECASE)
    predicted_value = match.group(1).capitalize() if match else None

    if debug:
        print(f"Valoare extrasă (după regex): {predicted_value}")
        print("--------------------")

    return predicted_value


def main():
    model_id = "google/flan-t5-large"
    model, tokenizer = load_model_and_tokenizer(model_id)
    if model is None or tokenizer is None:
        return

    text_generator = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer
    )

    train_data_path = "../data_sources/ro_stance_train_2.json"
    train_data = load_data(train_data_path)
    if train_data is None:
        print("Nu se poate continua, datele de antrenare nu au putut fi încărcate.")
        return

    print(f"\nPregătind exemplele few-shot din setul de antrenare ({1} exemplu per categorie, format EN)...")
    few_shot_examples_string = prepare_few_shot_examples(train_data, tokenizer, num_examples_per_category=1,
                                                         max_example_text_length=50)
    print("Exemple few-shot pregătite.")

    test_data_path = "../data_sources/ro_stance_test_2.json"
    test_data = load_data(test_data_path)
    if test_data is None:
        print("Nu se poate continua, datele de test nu au putut fi încărcate.")
        return

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "flan_t5_large_few_shot_1_example_predictions.json")

    mapping_model_output_to_ro = {
        "Pro": "Pro",
        "Against": "Contra",
        "Neutral": "Neutru"
    }

    results_to_save = []
    correct_predictions = 0
    total_predictions = 0
    invalid_predictions = 0

    total_annotations = sum(len(entry.get("annotations", [])) for entry in test_data)

    print(
        f"\nSe începe inferența cu modelul {model_id} (few-shot cu 1 exemplu, prompt zero-shot) pentru întregul fișier...")
    with tqdm(total=total_annotations, desc="Inferență articole", ncols=80) as pbar:
        for i, entry in enumerate(test_data):
            article_text = entry.get("article", "Text lipsă")
            annotations = entry.get("annotations", [])

            for annotation in annotations:
                target = annotation.get("target")
                true_label = annotation.get("label")

                if not target or not true_label:
                    pbar.update(1)
                    continue

                should_debug = (i < 5)

                predicted_label_raw = run_few_shot_inference(
                    text_generator, article_text, target, few_shot_examples_string, debug=should_debug,
                    tokenizer=tokenizer
                )
                total_predictions += 1

                mapped_prediction = mapping_model_output_to_ro.get(predicted_label_raw, "None")
                is_valid = mapped_prediction != "None"

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
    print(f"\n--- Rezumat Final (Few-Shot cu 1 exemplu, prompt zero-shot) ---")
    print(f"Total predicții: {total_predictions}")
    print(f"Predicții corecte: {correct_predictions}")
    print(f"Predicții invalide: {invalid_predictions}")
    print(f"Acuratețe totală: {accuracy:.2f}%")

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=4, ensure_ascii=False)
    print(f"Rezultatele au fost salvate în: {output_file_path}")


if __name__ == "__main__":
    main()