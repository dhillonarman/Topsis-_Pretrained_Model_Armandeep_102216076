# Topsis-_Pretrained_Model_Armandeep_102216076
Topsis to find best pretrained model for text generation

# 📌 TOPSIS Evaluation for Pretrained Text Generation Models

This project applies the **TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)** method to evaluate and rank pretrained text generation models based on multiple performance metrics.

---

## 📌 Overview

This repository provides a Python script to:

✅ Load pretrained **text generation models** (GPT-2, XLNet, and BLOOM) using the `transformers` library.  
✅ Generate text sequences using a common **prompt** ("Once upon a time,").  
✅ Evaluate the models using **three key metrics**:
   - **Perplexity**: Measures how well the model predicts the next word.
   - **Diversity**: Ratio of unique tokens in generated text.
   - **Inference Time**: Time taken to generate text sequences.
✅ Apply **TOPSIS** to rank the models based on these metrics.

---

## 📌 Prerequisites

Ensure you have **Python 3.x** installed along with the required dependencies:

```bash
pip install numpy pandas matplotlib torch transformers datasets scikit-learn
```

---

## 📌 Installation

Clone this repository and navigate to the project directory:

```bash
git clone https://github.com/your-repo/topsis-text-generation.git
cd topsis-text-generation
```

---

## 📌 Running the Project

Execute the script to run the evaluation:

```bash
python topsis_text_generation.py
```

This will generate the **TOPSIS ranking** and save the results to a CSV file.

---

## 📌 How It Works

### **1️⃣ Data Collection**
- The **wikitext-2-raw-v1** dataset is used for evaluation.  
- A **prompt** ("Once upon a time,") is used to generate text sequences.  

### **2️⃣ Model Evaluation**
For each model (**GPT-2, XLNet, and BLOOM**), we:  
✅ Generate **5 text sequences**.  
✅ Compute **perplexity, diversity, and inference time** for each sequence.  

### **3️⃣ TOPSIS Calculation**
- The **three metrics** are **normalized and weighted**.  
- The **ideal best and worst solutions** are calculated.  
- The **TOPSIS score** is computed to rank the models.  

---

## 📌 Evaluation Metrics

📍 **Perplexity**: Measures model confidence. **Lower is better**.  
📍 **Diversity**: Measures unique token ratio. **Higher is better**.  
📍 **Inference Time**: Measures response speed. **Lower is better**.  

---

## 📌 Example Output

After running the script, the output may look like this:

```plaintext
Model   | Perplexity | Diversity | Inference Time (s) | TOPSIS Score | Rank
--------|-----------|----------|------------------|--------------|------
GPT-2   | 32.5      | 0.68      | 0.45             | 0.742        | 2
XLNet   | 28.9      | 0.74      | 0.52             | 0.895        | 1
BLOOM   | 35.7      | 0.65      | 0.60             | 0.632        | 3
```

📌 The model with the **highest TOPSIS score** is ranked **best**.

---

## 📌 Results

📍 The final ranking is saved in **topsis_results.csv**.  
📍 A **bar chart** is generated for **visual analysis** of the rankings.  

---

## 📌 License

📜 This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.  

---

## 📌 Contributors

👥 Feel free to contribute by opening issues or submitting pull requests!  

🚀 **Happy Coding!** 🎯

