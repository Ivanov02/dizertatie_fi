import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import warnings
import re
import os
from tqdm import tqdm
import random

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

from env import token_hugging_face

MODEL_CACHE_DIR = None


def load_model_and_tokenizer(model_id: str, hf_token: str):
    print(f"Se încarcă modelul: {model_id}...")

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            token=hf_token,
            trust_remote_code=False
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=False)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        print("Modelul și tokenizer-ul au fost încărcate cu succes.")
        print(f"Modelul este plasat pe: {model.device}")
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
        num_examples_per_category: int = 1,
        max_example_text_length: int = 50  # Păstrat la 50 tokeni pentru exemple
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
                f"Stance: {example['label']}"
            )

    return "\n\n".join(few_shot_prompt_parts)


def run_few_shot_inference(
        text_generator: pipeline,
        article: str,
        target: str,
        few_shot_examples_str: str,
        tokenizer: AutoTokenizer = None,
        debug: bool = False
) -> str:

    MAX_ARTICLE_LENGTH_LLAMA = 4000

    truncated_article_text = tokenizer.decode(
        tokenizer.encode(article, max_length=MAX_ARTICLE_LENGTH_LLAMA, truncation=True),
        skip_special_tokens=True
    )

    full_prompt_content = (
        f"Analyze the following text and determine its stance towards the target. "
        f"Provide only the stance as 'Pro', 'Against', or 'Neutral'.\n\n"
        f"Examples:\n{few_shot_examples_str}\n\n"
        f"Text: \"{truncated_article_text}\"\n"
        f"Target: \"{target}\"\n\n"
        f"Stance:"
    )

    messages = [
        {"role": "system",
         "content": "You are a helpful assistant specialized in identifying the stance of a text towards a given target. Your response should be concise and only one of the following words: Pro, Against, Neutral."},
        {"role": "user", "content": full_prompt_content}
    ]

    prompt_with_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if tokenizer:
        prompt_length = len(tokenizer.encode(prompt_with_template, truncation=False))
        if prompt_length > 8192:
            print(f"Lungimea promptului este {prompt_length} tokeni (> 8192 - Llama max). "
                  f"Modelul va trunchia inputul. Reduceți MAX_ARTICLE_LENGTH_LLAMA, max_example_text_length sau num_examples_per_category.")
        elif prompt_length > 4000 and prompt_length <= 8192:  # Adjusted info threshold
            print(
                f"Lungimea promptului este {prompt_length} tokeni. Este în limita Llama, dar este un prompt lung.")

    if debug:
        print("\n--- FULL PROMPT (with chat template) ---")
        print(prompt_with_template)
        print("--------------------")

    outputs = text_generator(
        prompt_with_template,
        max_new_tokens=10,
        do_sample=False,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    raw_prediction = outputs[0]['generated_text'].strip()
    match = re.search(r"\b(pro|against|neutral)\b", raw_prediction, re.IGNORECASE)

    predicted_value = match.group(1).capitalize() if match else None

    if debug:
        print(f"Predicție brută model: \"{raw_prediction}\"")
        print(f"Valoare extrasă (după regex): {predicted_value}")
        print("--------------------")

    return predicted_value


def main():
    model_id = "meta-llama/Llama-3.2-3b-instruct"  # Keep Llama-3.2-3b-instruct

    hf_token = token_hugging_face

    model, tokenizer = load_model_and_tokenizer(model_id, hf_token)
    if model is None or tokenizer is None:
        return

    text_generator = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False
    )

    train_data_path = "../data_sources/ro_stance_train_2.json"
    train_data = load_data(train_data_path)
    if train_data is None:
        print("Nu se poate continua, datele de antrenare nu au putut fi încărcate.")
        return

    NUM_EXAMPLES_PER_CATEGORY = 1
    MAX_EXAMPLE_TEXT_LENGTH = 50

    print(
        f"\nPregătind exemplele few-shot din setul de antrenare ({NUM_EXAMPLES_PER_CATEGORY} exemplu per categorie, format EN)...")
    few_shot_examples_string = prepare_few_shot_examples(
        train_data,
        tokenizer,
        num_examples_per_category=NUM_EXAMPLES_PER_CATEGORY,
        max_example_text_length=MAX_EXAMPLE_TEXT_LENGTH
    )
    print("Exemple few-shot pregătite.")

    test_data_path = "../data_sources/ro_stance_test_2.json"
    test_data = load_data(test_data_path)
    if test_data is None:
        print("Nu se poate continua, datele de test nu au putut fi încărcate.")
        return

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "llama_3_2_3b_few_shot_1_example_full_dataset_predictions.json")

    mapping_model_output_to_ro = {
        "Pro": "Pro",
        "Against": "Contra",
        "Neutral": "Neutru"
    }

    results_to_save = []
    correct_predictions = 0
    total_predictions = 0
    invalid_predictions = 0

    total_annotations_full = sum(len(entry.get("annotations", [])) for entry in test_data)

    print(
        f"\nSe începe inferența cu modelul {model_id} (few-shot cu {NUM_EXAMPLES_PER_CATEGORY} exemplu per categorie) pentru ÎNTREGUL SET DE DATE...")
    with tqdm(total=total_annotations_full, desc="Inferență articole", ncols=80) as pbar:
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
    print(
        f"\n--- Rezumat Final (Few-Shot cu {NUM_EXAMPLES_PER_CATEGORY} exemplu per categorie, SET COMPLET DE DATE) ---")
    print(f"Total predicții: {total_predictions}")
    print(f"Predicții corecte: {correct_predictions}")
    f"Predicții invalide: {invalid_predictions}"
    print(f"Acuratețe totală: {accuracy:.2f}%")

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=4, ensure_ascii=False)
    print(f"Rezultatele au fost salvate în: {output_file_path}")


if __name__ == "__main__":
    main()