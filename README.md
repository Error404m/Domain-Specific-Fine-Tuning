# Neural Machine Translation: English to Dutch

A comparative study of encoder-decoder (mBART-50) and decoder-only (GPT-2+LoRA) architectures for machine translation.

---

## Table of Contents

- [Overview](#overview)
- [Why This Project Matters](#why-this-project-matters)
- [Architecture Comparison](#architecture-comparison)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Running the Code](#running-the-code)
- [Results](#results)
- [Key Findings](#key-findings)
- [What I Learned](#what-i-learned)
- [Future Improvements](#future-improvements)
- [References](#references)

---

## Overview

This project explores how different neural network architectures approach the translation problem. I trained and compared two fundamentally different models:

1. **mBART-50** (encoder-decoder): A specialized multilingual translation model
2. **GPT-2 with LoRA** (decoder-only): A general language model adapted for translation

The goal wasn't just to translate text it was to understand which architectural approach works better and why. Spoiler alert: the specialized translation architecture wins by a landslide, and the reasons are fascinating.

---

## Why This Project Matters

Machine translation is everywhere—from Google Translate to real-time video captions. But have you ever wondered what happens under the hood? 

When I started this project, I wanted to answer a specific question: Can we teach a general-purpose language model (like GPT-2) to translate by just showing it examples, or do we need architectures specifically designed for translation?

The answer has implications beyond just translation. It tells us something fundamental about whether specialized AI tools will always have a place, or if sufficiently large general models can do everything.

---

## Architecture Comparison

### mBART-50: The Specialist

Think of mBART-50 as a professional translator who reads the entire English sentence first, thinks about its meaning, and then writes it in Dutch. That's essentially what the encoder-decoder architecture does:

- **Encoder**: Reads the source sentence bidirectionally (sees all words at once)
- **Decoder**: Generates the translation word-by-word, constantly referring back to the source
- **Pre-trained on**: 50 languages with translation-specific objectives

```python
# How mBART processes translation
English → [Encoder] → Meaning Representation → [Decoder] → Dutch
```

### GPT-2 + LoRA: The Generalist

GPT-2 is like someone who's really good at continuing stories. You give it "Translate to Dutch: Hello" and hope it continues with the Dutch translation instead of just saying "world". We added LoRA (Low-Rank Adaptation) to make fine-tuning efficient:

- **No encoder**: Everything happens in one autoregressive pass
- **Prompt-based**: Relies on formatting like "Translate to Dutch: {text}\n\nTranslation:"
- **Pre-trained on**: English text continuation only

```python
# How GPT-2 processes translation
"Translate to Dutch: Hello\n\nTranslation:" → [Continue Text] → "Hallo"
```

**LoRA Magic**: Instead of updating all 124 million GPT-2 parameters, LoRA adds small trainable matrices to attention layers, reducing trainable parameters to just 2 million (~98% reduction) while maintaining learning capability.

---

## Dataset

I built a comprehensive training dataset by combining multiple high-quality sources:

### Training Data (50,000 sentence pairs)

| Source | Description | Why Include It |
|--------|-------------|----------------|
| **OPUS-100** | Diverse parallel corpus | Covers multiple domains (news, legal, subtitles) |
| **OPUS Books** | Literary translations | Rich, natural language with complex structures |
| **Tatoeba** | Community translations | Carefully validated by humans |
| **CCMatrix** | Web-crawled data | Large-scale, broad coverage |

**Data Quality Filters:**
- Sentence length: 5-150 words (removes noise and overly complex sentences)
- Language detection: Verified English-Dutch pairs
- Deduplication: Removed exact duplicates

### Test Data

1. **Software Domain** (`Dataset_Challenge_1.xlsx`)
   - Technical documentation translations
   - Software UI strings and error messages
   - Tests domain-specific vocabulary

2. **FLORES-200** (200 samples)
   - General domain text (news, conversations, narratives)
   - Standard benchmark for translation quality
   - Tests generalization ability

### Sample Training Pairs

```
[English]  "Click the submit button to proceed with the installation."
[Dutch]    "Klik op de verzendknop om door te gaan met de installatie."

[English]  "The weather forecast predicts rain throughout the weekend."
[Dutch]    "De weersvoorspelling voorspelt regen gedurende het weekend."

[English]  "Please enter your password to continue."
[Dutch]    "Voer uw wachtwoord in om door te gaan."
```

---

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, but CPU works)
- 16GB+ RAM
- ~10GB disk space for models and data

### Setup

```bash
# Clone the repository
git clone https://github.com/Error404m/Domain-Specific-Fine-Tuning.git
cd neural-translation-en-nl

# Install dependencies
pip install transformers datasets sentencepiece sacrebleu torch evaluate accelerate peft bitsandbytes openpyxl pandas matplotlib seaborn

# For specific versions (recommended)
pip install "datasets<4.0.0"
```

### Quick Verification

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
```

---

## Project Structure

```
neural-translation-en-nl/
│
├── mrityunjaya's_assignment.py    # Main training and evaluation script
├── Dataset_Challenge_1.xlsx       # Software domain test set
├── README.md                      # You are here!
│
├── outputs/                       # Generated during execution
│   ├── mbart_finetuned/          # Fine-tuned mBART model
│   ├── gpt2_lora_finetuned/      # Fine-tuned GPT-2 model
│   ├── evaluation_results.xlsx   # Numeric results
│   ├── software_domain_predictions.xlsx  # Translation outputs
│   └── comprehensive_results.png  # Visualization
│
├── docs/
│   └── project_report.tex         # LaTeX report
│
└── requirements.txt               # Dependency list
```

---

## Running the Code

### Full Pipeline (Training + Evaluation)

```bash
python mrityunjaya's_assignment.py
```

This will:
1. Load and preprocess 50,000 training pairs (~5 minutes)
2. Fine-tune mBART-50 (3 epochs, ~2 hours on GPU)
3. Fine-tune GPT-2 with LoRA (3 epochs, ~1.5 hours on GPU)
4. Evaluate both models on software and general domains (~10 minutes)
5. Generate visualizations and save results

**Expected Runtime**: 4-5 hours total on a single GPU (Tesla T4 or better)

### Google Colab (Recommended for Beginners)

```python
# In a Colab notebook
!git clone https://github.com/Error404m/Domain-Specific-Fine-Tuning.git
%cd Domain-Specific-Fine-Tuning
!python mrityunjaya's_assignment.py
```

Colab provides free GPU access, perfect for this project!

### Step-by-Step Execution

If you want to run stages separately:

```python
# 1. Load data only
train_data = load_production_corpus(max_samples=50000)
software_test = load_custom_dataset()

# 2. Train mBART only
# (uncomment mBART training section)

# 3. Train GPT-2 only
# (uncomment GPT-2 training section)

# 4. Evaluate only
# (requires trained models in output directories)
```

---
## Results

### Performance Comparison
![Results Visualization](https://raw.githubusercontent.com/Error404m/Domain-Specific-Fine-Tuning/refs/heads/main/comprehensive_results-2.png)

| Model | Dataset | BLEU ↑ | chrF ↑ | TER ↓ |
|-------|---------|--------|--------|-------|
| **mBART-50** | Software Domain | **19.26** | **44.00** | **66.19** |
| GPT-2+LoRA | Software Domain | 0.57 | 11.59 | 304.70 |
| **mBART-50** | General (FLORES) | **18.55** | **50.08** | **65.43** |
| GPT-2+LoRA | General (FLORES) | 1.50 | 16.90 | 210.69 |

**Metric Guide:**
- **BLEU** (0-100): Measures n-gram overlap with reference. Higher = better. 20+ is decent, 40+ is excellent.
- **chrF** (0-100): Character-level F-score. More forgiving than BLEU. 40+ is good.
- **TER** (0-∞): Edit distance percentage. Lower = better. 60-70 is acceptable.

### What These Numbers Mean

**mBART-50's Performance:**
- Produces mostly correct translations with minor stylistic variations
- BLEU ~19 is solid for this dataset size and domain
- chrF ~44-50 shows good vocabulary and structure
- TER ~65 means translations need moderate editing to match reference exactly

**GPT-2+LoRA's Performance:**
- BLEU near 0 indicates fundamental translation failure
- Produces Dutch-English hybrids rather than proper translations
- chrF ~12-17 shows some character-level patterns learned
- TER >200 means output is longer than reference and mostly wrong

### Visual Analysis

The generated visualization (`comprehensive_results.png`) shows three key insights:

1. **BLEU scores**: mBART dominates with scores 20-30× higher
2. **chrF scores**: mBART maintains 3-4× advantage even in forgiving metrics
3. **TER scores**: GPT-2's scores exceed 200%, indicating structural failure

---

##  Key Findings

### 1. Architecture Matters More Than You Think

I went into this expecting GPT-2 to struggle but thinking it would at least be competitive. It wasn't. The specialized encoder-decoder architecture provides advantages that parameter-efficient fine-tuning can't overcome:

- **Bidirectional encoding**: mBART sees the entire source sentence before generating anything
- **Cross-attention**: The decoder can "look at" different source words for different target words
- **Separation of concerns**: Encoding and generation are distinct processes

### 2. Pre-training Shapes Capabilities

mBART was pre-trained on 50 languages with translation-like objectives. GPT-2 was pre-trained on English text completion. This prior knowledge creates a massive head start that's hard to overcome with task-specific fine-tuning.

### 3. Parameter Efficiency Has Limits

LoRA reduced trainable parameters from 124M to 2M—a 98% reduction. This is amazing for efficiency, but it doesn't make the model magically good at translation. You can't train 2M parameters to override 124M parameters of "wrong" prior knowledge.

### 4. Domain Doesn't Matter Much for Relative Performance

Both models did slightly better on general text than software documentation, but the gap between them remained consistent. This suggests the performance difference is fundamental, not domain-specific.

### 5. Prompting Isn't a Universal Solver

I used a straightforward prompt format: "Translate to Dutch: {text}\n\nTranslation:". More sophisticated prompting might help, but it won't close a 20× performance gap. The model needs to fundamentally understand the translation task.

---

## What I Learned

### Technical Insights

1. **Metric Selection Matters**: Looking at multiple metrics revealed different failure modes. GPT-2's chrF scores being higher than BLEU showed it learned some patterns, just not coherent translation.

2. **Batch Size vs. Quality**: Smaller batches (4-8) worked fine. Bigger isn't always better, especially with good data.

3. **Warmup is Essential**: Starting with a low learning rate and ramping up prevented early training instability in both models.

4. **Mixed Precision Training**: FP16 halved memory usage with negligible quality impact. Always use it if your GPU supports it.

### Practical Lessons

1. **Start with the Right Architecture**: Don't try to force a square peg into a round hole. If your task has structure (like translation), use an architecture that matches it.

2. **Pre-training Isn't Everything, But It's A Lot**: A well-pre-trained specialist beats a poorly-pre-trained generalist, even with more fine-tuning data.

3. **Evaluate on Multiple Metrics**: BLEU alone would've hidden some of GPT-2's partial successes. Multiple metrics give a fuller picture.

4. **Real Examples Beat Numbers**: The quantitative results were clear, but showing actual translations made the difference visceral.

### Research Questions for Future Work

- Would GPT-3 scale models do better? Does sheer size overcome architectural mismatch?
- Can we design hybrid architectures that combine encoder-decoder strengths with decoder-only flexibility?
- What's the minimal architecture needed for good translation? Can we prune mBART while keeping performance?

---

## Limitations and Challenges

### Computational Constraints

Training large models is expensive. I worked within a single GPU budget, which meant:
- Smaller batch sizes than ideal (4-8 vs. 32-64)
- Limited hyperparameter search (couldn't try many learning rates)
- Shorter training (3 epochs vs. 10+ in production systems)

### Dataset Size

50,000 sentence pairs is respectable but not huge. State-of-the-art systems train on millions of pairs. More data would likely:
- Improve both models' absolute performance
- Maintain the relative gap (mBART would still dominate)
- Allow longer training without overfitting

### Evaluation Limitations

Automatic metrics (BLEU, chrF, TER) don't capture everything:
- A translation can be correct but score poorly if it uses different phrasing
- Cultural and contextual nuances are invisible to these metrics
- Human evaluation would provide additional insight

### GPT-2 Prompt Engineering

I used a simple, straightforward prompt. More sophisticated approaches might help:
- Few-shot examples in the prompt
- Chain-of-thought reasoning
- Instruction fine-tuning before translation fine-tuning

However, these would add complexity and might not close the fundamental gap.

---

## Future Improvements

### Short Term (Achievable with More Resources)

1. **Larger Models**: Test GPT-3.5 or LLaMA-2 to see if scale changes conclusions
2. **More Data**: Train on 500K pairs to reduce data constraint
3. **Human Evaluation**: Have native Dutch speakers rate a sample of translations
4. **Error Analysis**: Categorize errors (word order, vocabulary, grammar) systematically
5. **Better Prompting**: Experiment with few-shot and instruction-tuned approaches

### Medium Term (Research Directions)

1. **Hybrid Architecture**: Add an encoder to GPT-2 style models
2. **Domain Adaptation**: Fine-tune further on medical/legal text
3. **Interactive Learning**: Allow users to provide feedback and retrain
4. **Multilingual Extension**: Add more language pairs to compare
5. **Interpretability**: Visualize attention patterns to understand what each model learns

### Long Term (Open Questions)

1. **Architectural Minimalism**: What's the simplest architecture that works?
2. **Generalist vs. Specialist**: Will future models make specialists obsolete?
3. **Low-Resource Translation**: How do these approaches work with 1000 training pairs?
4. **Continuous Learning**: Can models improve from user corrections over time?

---

## References

### Papers
- Vaswani et al. (2017) - "Attention is All You Need" (Transformer architecture)
- Liu et al. (2020) - "Multilingual Denoising Pre-training for NMT" (mBART)
- Radford et al. (2019) - "Language Models are Unsupervised Multitask Learners" (GPT-2)
- Hu et al. (2021) - "LoRA: Low-Rank Adaptation of Large Language Models"

### Datasets
- OPUS Project: https://opus.nlpl.eu/
- FLORES: https://github.com/facebookresearch/flores
- Tatoeba: https://tatoeba.org/

### Tools & Libraries
- Hugging Face Transformers: https://huggingface.co/docs/transformers
- PEFT (Parameter-Efficient Fine-Tuning): https://github.com/huggingface/peft
- SacreBLEU: https://github.com/mjpost/sacrebleu

---

## Author

**Mrityunjaya Tiwari**


---

*Last updated: 6-January 2026*
