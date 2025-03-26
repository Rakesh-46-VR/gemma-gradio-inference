# Gemma Model Interaction - Gradio Interface

## Overview

This project provides an interactive Gradio-based web interface for loading and generating text using Gemma models from Keras Hub. It supports multiple models and ensures efficient memory management when switching between models.

## Features

- Load different Gemma models interactively
- Generate text based on user-provided prompts
- Adjustable parameters for text generation
- Automatic GPU memory cleanup before loading a new model

## Requirements

Ensure you have the following installed:

- Python 3.8+

You can install dependencies using:

```sh
pip install -r req.txt
```

## Environment Variables

This project requires API credentials for Kaggle. Ensure you have a `.env` file in the project root with the following:

```sh
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
```

## How It Works

1. **Model Selection & Loading**: Select a model from the dropdown and click "Load Model."
2. **Text Generation**: Enter a prompt, adjust parameters, and click "Generate" to get a response.

## Running the Application

Run the script using:

```sh
python infer_gradio.py
```

This will launch the Gradio interface at `http://localhost:7860/`.

## Available Models

| Model Name         | Preset Name               |
| ------------------ | ------------------------- |
| Gemma2-2B          | `gemma2_instruct_2b_en` |
| Gemma2-2B-Instruct | `gemma2_2b_en`          |
| Code-Gemma         | `code_gemma_2b_en`      |

## Memory Management

To ensure optimal GPU memory usage:

- The old model is deleted before loading a new one.
- `torch.cuda.empty_cache()` and `gc.collect()` are called to free memory.
- `torch.cuda.synchronize()` ensures memory release before new allocations.
