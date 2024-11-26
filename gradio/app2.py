import gradio as gr
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM

# Cargar el modelo BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Cargar modelo para generación de poemas (cambiamos a un modelo más adecuado)
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
poem_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def generate_description(image):
    if image is None:
        return "No se ha proporcionado imagen"
    
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    return description

def reset():
    return ""

def generate_story_prompt(desc1, desc2, desc3):
    # Validacion de descripciones
    if not all([desc1, desc2, desc3]):
        return "Por favor, describe las tres imágenes primero"
    
    # Prompt claro
    prompt = f"""Escribe un poema corto de 4 líneas que conecte estas tres imágenes de manera creativa:
    Imagen 1: {desc1}
    Imagen 2: {desc2}
    Imagen 3: {desc3}

    El poema debe ser en español, rimado y coherente, usando elementos de las descripciones."""
    
    return prompt

def generate_poem(prompt, desc1, desc2, desc3):
    # Validar que el prompt no esté vacío
    if not prompt or "Por favor" in prompt:
        return "Primero genera un prompt válido"

    poem_instructions = """Siguiendo el prompt, crea un poema corto de 4 líneas en español, con rima y usando los elementos descritos:"""
    full_prompt = poem_instructions + prompt
    
    # Generar poema
    inputs = tokenizer.encode(full_prompt, return_tensors="pt")
    outputs = poem_model.generate(
        inputs, 
        max_length=200,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        do_sample=True
    )

    raw_poem = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
 
    poem_lines = raw_poem.split('\n')
    cleaned_poem_lines = [line.strip() for line in poem_lines if line.strip()]
 
    poem_candidates = [line for line in cleaned_poem_lines if len(line.split()) > 3]

    if len(poem_candidates) < 4:
        poem = generar_poema_manual(desc1, desc2, desc3)
        return poem
    

    final_poem = '\n'.join(poem_candidates[:4])
    return final_poem

def generar_poema_manual(desc1, desc2, desc3):
    """Función de respaldo para generar un poema si la generación automática falla"""

    elementos = [desc1, desc2, desc3]
    

    plantillas = [
        f"""En {elementos[0]}, un paisaje de ensueño,
{elementos[1]} brilla con su empeño.
{elementos[2]} nos invita a soñar,
Un momento único para recordar.""",
        
        f"""Viajando por {elementos[0]} con alegría,
{elementos[1]} ilumina nuestra día.
{elementos[2]} completa la escena,
Una historia que el corazón llena.""",
        
        f"""Contemplando {elementos[0]} con emoción,
{elementos[1]} despierta mi pasión.
{elementos[2]} corona la aventura,
Un poema de bella escritura."""
    ]
    
    # Elegir una plantilla al azar
    return plantillas[torch.randint(0, len(plantillas), (1,)).item()]

with gr.Blocks() as demo:
    descriptions = []

    for i in range(3):
        with gr.Row():
            with gr.Column():
                image = gr.Image(label=f"Imagen {i+1}")
                text = gr.Textbox(label=f"Descripción {i+1}")
                descriptions.append(text)
                with gr.Row():
                    reset_button = gr.Button("Resetear")
                    reset_button.click(fn=reset, inputs=[], outputs=[text])
                    submit_button = gr.Button("Describir")
                    submit_button.click(fn=generate_description, inputs=[image], outputs=[text])

    with gr.Row():
        prompt_box = gr.Textbox(label="Prompt para poema", interactive=False)
        poem_box = gr.Textbox(label="Poema generado", interactive=False)

    with gr.Row():
        generate_prompt_button = gr.Button("Generar Prompt")
        generate_prompt_button.click(
            fn=generate_story_prompt, 
            inputs=descriptions, 
            outputs=[prompt_box]
        )

        generate_poem_button = gr.Button("Generar Poema")
        generate_poem_button.click(
            fn=generate_poem, 
            inputs=[prompt_box] + descriptions,  # Añadir las descripciones como entradas
            outputs=[poem_box]
        )

demo.launch()
