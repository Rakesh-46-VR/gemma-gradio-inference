import gradio as gr
import os
import keras_hub
import inspect
import torch
import gc


os.environ["KAGGLE_USERNAME"] = os.getenv('KAGGLE_USERNAME')
os.environ["KAGGLE_KEY"] = os.getenv('KAGGLE_KEY')

os.environ["KERAS_BACKEND"] = "torch"  # Or "tensorflow" or "torch".
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"

model = None  # Global model variable

# Dictionary of Gemma models
MODELS = {
    "Gemma2-2B": "gemma2_instruct_2b_en",
    "Gemma2-2B-Instruct": "gemma2_2b_en",
    "Code-Gemma" : "code_gemma_2b_en"
}

def load_model(preset_name):
    """
    Loads a new model from the given preset. Before loading,
    it properly deletes the old model, clears the GPU memory, 
    and ensures a fresh load.
    """
    global model
    if 'model' in globals() and model is not None:
        print("Unloading existing model...")
        
        # Delete model and free memory
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    print(f"Loading model {preset_name}...")
    
    # Force synchronization before reloading to ensure complete memory release
    torch.cuda.synchronize()
    
    # Load new model
    model = keras_hub.models.GemmaCausalLM.from_preset(preset_name)
    
    return f"Model '{preset_name}' loaded successfully."
def generate_text(prompt, max_length, temperature, top_p):
    """
    Generate text using the loaded model with the specified parameters.
    """
    global model
    if model is None:
        return "No model is loaded. Please load a model first."
    if not prompt:
        return "Please enter a prompt."
    
    
    output = model.generate(
        prompt,
        max_length=max_length,
        strip_prompt=True
    )
    return output

with gr.Blocks() as demo:
    gr.Markdown("## Gemma Model Interaction")

    # Step 1: Model selection and loading
    gr.Markdown("### Step 1: Choose and Load Your Model")
    model_name_dropdown = gr.Dropdown(
        label="Select Model",
        choices=list(MODELS.keys()),
        value="Gemma-2B"
    )
    load_button = gr.Button("Load Model")
    load_status = gr.Textbox(label="Model Loading Status", interactive=False)

    # Step 2: Text generation UI
    gr.Markdown("### Step 2: Generate Text from the Loaded Model")
    prompt_input = gr.Textbox(label="Enter Prompt")
    max_length_slider = gr.Slider(
        label="Max Output Tokens", minimum=1, maximum=512, value=30, step=1
    )
    generate_button = gr.Button("Generate")
    output_box = gr.Textbox(label="Generated Response")

    # Bind the load_model function using a lambda to map the dropdown selection to its preset name.
    load_button.click(
        fn=lambda selected: load_model(MODELS[selected]),
        inputs=model_name_dropdown,
        outputs=load_status
    )

    # Bind the generate_text function.
    generate_button.click(
        fn=generate_text,
        inputs=[prompt_input, max_length_slider],
        outputs=output_box
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
