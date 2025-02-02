import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time
from math import exp

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_perplexity(model, tokenizer, texts):
    """Compute perplexity for causal models."""
    model.eval()
    losses = []
    with torch.no_grad():
        for text in texts:
            if len(text.strip()) == 0:
                continue
            encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = encodings.input_ids
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            losses.append(loss.item())
    return exp(np.mean(losses)) if losses else float('inf')

def compute_diversity(generated_texts):
    """Diversity: unique tokens ratio."""
    all_tokens = [word for text in generated_texts for word in text.split()]
    return len(set(all_tokens)) / (len(all_tokens) + 1e-5)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
eval_texts = dataset["text"][:100]

prompt_text = "Once upon a time,"

gen_kwargs_causal = {
    "max_length": 50,
    "do_sample": True,
    "num_return_sequences": 1,
    "no_repeat_ngram_size": 2
}

gen_kwargs_xlnet = {
    "max_length": 50,
    "top_k": 50,
    "top_p": 0.95,
    "temperature": 0.7,
    "num_return_sequences": 1,
    "no_repeat_ngram_size": 3,
    "do_sample": True
}

models = {
    "GPT-2": {"type": "causal", "model_name": "gpt2"},
    "XLNet": {"type": "causal", "model_name": "xlnet-base-cased"},
    "Bloom": {"type": "causal", "model_name": "bigscience/bloom-560m"}
}

metrics = {}
num_generations = 5

for model_name, info in models.items():
    print(f"\nEvaluating {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(info["model_name"], trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(info["model_name"])

    generated_texts, gen_times = [], []

    for i in range(num_generations):
        start = time.time()
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
        output = model.generate(input_ids, **(gen_kwargs_xlnet if model_name == "XLNet" else gen_kwargs_causal))
        gen_text = tokenizer.decode(output[0], skip_special_tokens=True)
        elapsed = time.time() - start

        gen_times.append(elapsed)
        generated_texts.append(gen_text)
        print(f"Run {i+1}: {gen_text}")

    avg_inference_time = np.mean(gen_times)
    ppl = compute_perplexity(model, tokenizer, generated_texts)
    diversity = compute_diversity(generated_texts)

    metrics[model_name] = [ppl, diversity, avg_inference_time]

data = np.array(list(metrics.values()))

data = np.where(np.isnan(data), np.nanmean(data, axis=0, keepdims=True), data)

weights = np.array([0.3, 0.4, 0.3])
impacts = np.array(["-", "+", "-"])

norm_data = data / np.sqrt((data**2).sum(axis=0))
weighted_data = norm_data * weights

ideal_best = np.where(impacts == "+", weighted_data.max(axis=0), weighted_data.min(axis=0))
ideal_worst = np.where(impacts == "+", weighted_data.min(axis=0), weighted_data.max(axis=0))

dist_best = np.sqrt(((weighted_data - ideal_best) ** 2).sum(axis=1))
dist_worst = np.sqrt(((weighted_data - ideal_worst) ** 2).sum(axis=1))

topsis_scores = dist_worst / (dist_best + dist_worst)

results_df = pd.DataFrame({
    'Model': list(metrics.keys()),
    'Perplexity': data[:, 0],
    'Diversity': data[:, 1],
    'Inference Time (s)': data[:, 2],
    'TOPSIS Score': topsis_scores
})

results_df["Rank"] = results_df["TOPSIS Score"].rank(ascending=False, method="dense").astype(int)

print("\nEvaluation Results:")
print(results_df)

results_df.to_csv("topsis_results.csv", index=False)
print("\nResults saved to 'topsis_results.csv'.")

plt.figure(figsize=(8, 5))
plt.bar(results_df['Model'], results_df['TOPSIS Score'], color=['blue', 'red', 'green'])
plt.xlabel("Models")
plt.ylabel("TOPSIS Score")
plt.title("TOPSIS Ranking of Text Generation Models")
plt.show()
