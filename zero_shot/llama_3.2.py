import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import warnings
import re
import os
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

from env import token_hugging_face

def load_model_and_tokenizer(model_id, hf_token):
    print(f"Loading model: {model_id}...")

    # Configuration for 8-bit quantization with CPU offload enabled
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        # Permite ca modulele care nu încap pe GPU să fie plasate pe CPU în format float32
        llm_int8_enable_fp32_cpu_offload=True
    )


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

    print("Model and tokenizer loaded successfully.")
    print(f"Model is placed on: {model.device}")
    return model, tokenizer

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_zero_shot_inference(text_generator, article, target, tokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specialized in identifying the stance of a text towards a given target. Your response should be concise and only one of the following words: Pro, Against, Neutral."},
        {"role": "user", "content": f"Analyze the following text and determine its stance towards the target. Provide only the stance as 'Pro', 'Against', or 'Neutral'.\n\nText: \"{article}\"\nTarget: \"{target}\"\n\nStance:"}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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
    model_id = "meta-llama/Llama-3.2-3b-instruct"

    model, tokenizer = load_model_and_tokenizer(model_id, token_hugging_face)

    text_generator = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False
    )

    test_data = load_data("../data_sources/ro_stance_test_2.json")

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    # Nume fișier pentru rezultate cu 8-bit și offload, pentru claritate
    output_file_path = os.path.join(output_dir, "llama_3_2_3b_8bit_offload_predictions_full_dataset_eng_ro.json")

    mapping_eng_to_ro = {
        "Pro": "Pro",
        "Against": "Contra",
        "Neutral": "Neutru"
    }

    results_to_save = []
    correct_predictions = 0
    total_predictions = 0
    invalid_predictions = 0

    print(f"\nStarting inference with model {model_id} on the entire dataset (8-bit quantization with CPU offload)...")
    total_annotations = sum(len(entry.get("annotations", [])) for entry in test_data)

    with tqdm(total=total_annotations, desc="Inference on dataset", ncols=80) as pbar:
        for entry in test_data:
            article_text = entry.get("article", "Text missing")
            annotations = entry.get("annotations", [])

            for annotation in annotations:
                target = annotation.get("target")
                true_label = annotation.get("label")

                if not target or not true_label:
                    pbar.update(1)
                    continue

                predicted_label = run_zero_shot_inference(text_generator, article_text, target, tokenizer)
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
    print(f"\n--- Final Summary (Full Dataset, 8-bit Quantization with CPU offload) ---")
    print(f"Total predictions: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Invalid predictions: {invalid_predictions}")
    print(f"Total Accuracy: {accuracy:.2f}%")

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=4, ensure_ascii=False)
    print(f"Results saved to: {output_file_path}")

if __name__ == "__main__":
    main()