# Fine-Tuning a Language Model for Quantitative Reasoning

This project explores advanced techniques for fine-tuning a Large Language Model (SmolLM2-360M) to perform accurate unit conversions. The goal is to enhance the model's quantitative reasoning abilities through prompt engineering, supervised fine-tuning (SFT), and reinforcement learning (RL).

## Project Overview

The project is divided into several parts, each building on the last to improve the model's performance:

1.  **Core Inference Engine (`base_llm.py`):** Implementation of batched and non-batched text generation functions using PyTorch and Hugging Face Transformers. This provides a deeper understanding of the model's inference mechanics and offers a significant performance boost through batching.

2.  **Chain-of-Thought Prompting (`cot.py`):** Using sophisticated prompt engineering with in-context learning to guide the base model. This enables the LLM to perform multi-step reasoning to solve conversion problems without any fine-tuning.

3.  **Supervised Fine-Tuning (`sft.py`):** Employing Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA) to efficiently fine-tune the model on a specialized dataset of unit conversions. This significantly improves task-specific performance while keeping training costs low.

4.  **Rejection-sampling Fine-Tuning (`datagen.py`, `rft.py`):** Implementing RFT, a reinforcement learning technique to further refine the model. This involves generating a high-quality dataset of reasoning chains and using it to train a model that not only answers correctly but also "thinks" step-by-step.



## Technologies Used

* **Python 3.12**
* **PyTorch**
* **Hugging Face:**
    * Transformers (for models and tokenizers)
    * PEFT (for LoRA)
    * Accelerate
* **TensorBoard** for logging



## Setup

1.  **Create a Conda environment:**
    ```bash
    conda create --name llm_unit_converter python=3.12 -y
    conda activate llm_unit_converter
    ```

2.  **Install PyTorch:**
    Follow the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/).

3.  **Install project dependencies:**
    ```bash
    pip install -r requirements.txt
    ```



## Usage

You can test the different models and training scripts using the following commands.

### Test the Base and CoT Models

To test the base generation or the Chain-of-Thought model:
```bash
### Test base model generation
python -m llm_unit_converter.base_llm test
```

### Test Chain-of-Thought model
```bash
python -m llm_unit_converter.cot test
```



## Train a New Model

To train the Supervised Fine-Tuning (SFT) or Rejection-sampling Fine-Tuning (RFT) models:


### Train the SFT model
```bash
python -m llm_unit_converter.sft train --output_dir ./sft_model_checkpoint
```

### First, generate data for RFT
```bash
python -m llm_unit_converter.datagen generate_dataset --output_json ./data/rft_data.json
```

###  Then, train the RFT model
```bash
python -m llm_unit_converter.rft train --output_dir ./rft_model_checkpoint
```



## Load a Pre-Trained Model

The project is structured to load fine-tuned models (LoRA adapters) from specific directories (sft_model and rft_model). Make sure your trained checkpoints are saved in or moved to these locations within the llm_unit_converter directory.


### Test a trained SFT model
```bash
python -m llm_unit_converter.sft test --ckpt_path ./sft_model_checkpoint
```

### Test a trained RFT model
```bash
python -m llm_unit_converter.rft test --ckpt_path ./rft_model_checkpoint
```
