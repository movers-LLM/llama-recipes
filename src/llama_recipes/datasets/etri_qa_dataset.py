import datasets

def get_preprocessed_etri_qa_pair(dataset_config, tokenizer, split, is_train_stage=True):
    dataset = datasets.load_dataset("Movers-AI/etri-qa-nodup", split='train')

    prompt = "Answer the question based on the context.\nQuestion:\n{question}\n---\nContext:\n{context}\n---\nAnswer:\n"

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(question=(sample['question'] if split=='train' else sample['question_paraphrase'][0])),
            "answer": sample['answer'],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    
    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        answer = tokenizer.encode(sample["answer"] +  tokenizer.eos_token, add_special_tokens=False)

        if is_train_stage:
            sample = {
                "input_ids": prompt + answer,
                "attention_mask" : [1] * (len(prompt) + len(answer)),
                "labels": [-100] * len(prompt) + answer,
            }
        else:
            sample = {
                "input_ids": prompt,
                "attention_mask": [1] * len(prompt),
                "labels": answer
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    return dataset

def get_preprocessed_etri_clm(dataset_config, tokenizer, split, is_train_stage=True):
    raise NotImplementedError("get_preprocessed_etri_clm is not implemented yet.")

if __name__ == '__main__':
    pass