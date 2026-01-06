# Domain-Specific Fine-Tuning for Machine Translation

English to Dutch

---

## Project Overview

This project focuses on improving machine translation quality for software-related text.

Instead of relying only on general-purpose translation models, this work fine-tunes models using software domain data so they can better handle technical terms, commands, and structured language.

Two different model architectures are trained and compared to understand how domain-specific fine-tuning impacts translation quality.

---

## Objectives

The main objectives of this project are:

* Fine-tune machine translation models using software domain data
* Compare encoder-decoder and decoder-only architectures
* Evaluate performance on both general and software-specific datasets
* Analyze improvements gained through domain adaptation

---

## Models Used

### Encoder-Decoder Model

* mBART-50 multilingual model
* Supports English and Dutch
* Designed for sequence-to-sequence translation tasks

### Decoder-Only Model

* GPT-2
* Fine-tuned using LoRA
* Efficient training with fewer trainable parameters

---

## Datasets

### Training Dataset

* Parallel English–Dutch data from WMT 2016 and similar sources
* Used to adapt models toward the software domain

### Evaluation Datasets

General domain evaluation

* FLORES devtest dataset

Software domain evaluation

* Dataset_Challenge_1.xlsx
* Contains English software-related sentences and corresponding Dutch translations

---

## Repository Structure

```
├── Notebook.ipynb
├── Dataset_Challenge_1.xlsx
├── README.md
```

The notebook contains the complete pipeline including data loading, fine-tuning, and evaluation.

---

## Workflow

```
Data Loading
     ↓
Preprocessing and Cleaning
     ↓
Tokenization
     ↓
Model Fine-Tuning
     ↓
Evaluation
     ↓
Comparison of Results
```

---

## How to Run the Project

Follow these steps carefully.

1. Open Google Colab
2. Upload the notebook file
3. Upload Dataset_Challenge_1.xlsx
4. Enable GPU from runtime settings
5. Run the dependency installation cell
6. Execute all cells from top to bottom
7. Wait until training finishes
8. View evaluation outputs at the end

Training time depends on the GPU provided by Colab.

---

## Colab Screenshots

Below are example screenshots from the Google Colab environment.
You can replace these image files with your actual screenshots.

### Dataset Upload

![Dataset Upload](screenshots/dataset_upload.png)

---

### GPU Configuration

![GPU Configuration](screenshots/gpu_settings.png)

---

### Training in Progress

![Training Process](screenshots/training_progress.png)

---

### Evaluation Results

![Evaluation Results](screenshots/evaluation_results.png)

---

## Evaluation Method

Both models are evaluated using two perspectives.

General domain evaluation measures how well the model translates everyday language.

Software domain evaluation measures how well the model translates technical and software-related sentences.

BLEU score and other standard translation metrics are used to compare results.

---

## Results and Observations

The encoder-decoder model performs strongly on structured translation tasks.

The decoder-only model with LoRA adapts efficiently while using fewer parameters.

Both models show clear improvement on software domain data after fine-tuning.

Domain-specific training significantly improves translation accuracy for technical text.

---

## Conclusion

This project shows that domain-specific fine-tuning is highly effective for machine translation.

Software-focused data helps models understand technical language better.

Both encoder-decoder and decoder-only approaches can be successfully adapted when trained correctly.
