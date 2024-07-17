import datasets

def get_preprocessed_aihub_translation(dataset_config, tokenizer, split, is_train_stage=True):
    dataset = datasets.load_dataset("Movers-AI/aihub-ko-en-ict-translation", split=split)

    prompt = (
        f"Translate this sentence to Korean:\n{{en}}\n---\nTranslation:\n"
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(en=sample["en"]),
            "translation": sample["ko"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    
    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        translation = tokenizer.encode(sample["translation"] +  tokenizer.eos_token, add_special_tokens=False)

        if is_train_stage:
            sample = {
                "input_ids": prompt + translation,
                "attention_mask" : [1] * (len(prompt) + len(translation)),
                "labels": [-100] * len(prompt) + translation,
            }
        else:
            sample = {
                "input_ids": prompt,
                "attention_mask": [1] * len(prompt),
                "labels": translation
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    return dataset