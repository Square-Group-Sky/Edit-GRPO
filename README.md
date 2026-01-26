

# Edit-GRPO: Enhancing Generalization in Model Editing for LLMs via Reinforcement Learning

This repository contains the implementation of **Edit-GRPO**, a reinforcement learning-based refinement framework designed to solve the **knowledge isolation** problem in Large Language Models (LLMs) after knowledge editing.

Unlike traditional editors that focus on initial injection, Edit-GRPO serves as a **Post-Editing Refinement** stage. It leverages **Anchor-Generalization Training (AGT)** and **Policy Gradient Alignment Advantage (PGAA)** to smooth the semantic manifold and enhance generalization.

![](C:\Users\29923\Desktop\ICML2026投稿文件\main-structure.jpg)

## 🛠 Prerequisites: The Two-Stage Paradigm

**Important:** Edit-GRPO is not a standalone knowledge injector. It is designed to refine a model that has already undergone an initial edit.

1. **Stage 1 (Model Edit):** Use a standard editor (e.g., ROME, MEMIT, or WISE) to inject a new fact into the model.
2. **Stage 2 (RL):** Run this framework using the edited model as the starting policy $\pi_{\theta_{edit}}$ to align the local manifold.

------

## 🚀 Environment Setup

```
We recommend using `Conda` to manage your environment. The setup is divided into two stages:

### Stage 1: Model Editing
For the initial knowledge editing phase, we rely on the `EasyEdit` framework. Please follow their official instructions for environment setup.

### Stage 2: Reinforcement Learning
For the GRPO-based alignment refinement, we build upon the `verl` framework. Please install the required dependencies according to their documentation.
```

------

## 📖 How to Run

### 1. Prepare the Datasets

- **Dataset (Hugging Face Hub)**: The official KnowEdit benchmark can be found at **[`https://huggingface.co/datasets/zjunlp/KnowEdit`](https://huggingface.co/datasets/zjunlp/KnowEdit)**. This is the primary source for the data splits (`CounterFact`, `ZsRE`, `WikiRecent`) used in experiments.
- **Code Repository (GitHub)**: The corresponding knowledge editing framework, which includes data loaders, editing methods, and evaluation scripts, is located at **[`https://github.com/zjunlp/EasyEdit`](https://github.com/zjunlp/EasyEdit)**.

### 2. Initial Model Editing

Before running Edit-GRPO, generate your edited checkpoint. You can use **[EasyEdit](https://github.com/zjunlp/EasyEdit)** to create a model edited with the method you like.

### 3. Running Edit-GRPO Refinement

We provide a complete training script for the Qwen3-8B model. You can run it with:

bash

```
bash ./examples/edit_grpo_trainer/run_qwen3-8b.sh
```

## 🔗 References & Credits

This project stands on the shoulders of giants in the model editing and RL community:

- **[EasyEdit](https://github.com/zjunlp/EasyEdit):** An easy-to-use framework for editing Large Language Models.
- **[verl](https://github.com/volcengine/verl):** A high-performance reinforcement learning library for LLMs (used for the GRPO implementation).