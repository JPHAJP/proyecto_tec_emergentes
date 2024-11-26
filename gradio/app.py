from PIL import Image
import numpy as np
import openai
import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from inference_sdk import InferenceHTTPClient

# Configura tu clave de API de OpenAI
openai.api_key = "APIKEY"
CLIENT = InferenceHTTPClient(
    api_url="https://classify.roboflow.com",
    api_key="KEYROBOFLOW"  # Tu API Key
)

# Configura BLIP para la descripción de imágenes
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Variable global para la predicción de emociones
emotion_prediction = ""

# Función para enviar la imagen y obtener resultados
def send_image(image):
    global emotion_prediction  # Permite actualizar la predicción globalmente
    result = CLIENT.infer(image, model_id="expression-bivfq-pugqb/1")
    predicted_classes = result["predicted_classes"]  # Extrae las clases predichas
    emotion_prediction = predicted_classes  # Almacena la predicción globalmente
    return image, f"Predicción: {predicted_classes}"

# Función para resetear la imagen del input
def reset_image():
    global emotion_prediction
    emotion_prediction = ""  # Reinicia la predicción global
    return None, "Predicción: ---"

# Función para generar descripciones de imágenes con BLIP
def generate_description(image):
    if image is None:
        return "No se ha proporcionado imagen."
    
    # Convertir la imagen de numpy.ndarray a PIL.Image si es necesario
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Generar descripción con BLIP
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs)
    description = processor.decode(outputs[0], skip_special_tokens=True)
    return description

# Función para generar un poema con OpenAI
def generate_poem(desc1, desc2, desc3):
    global emotion_prediction  # Usa la predicción global de emociones
    # Validar que todas las descripciones estén completas
    if not all([desc1, desc2, desc3]) or not emotion_prediction:
        return "Por favor, describe las tres imágenes y asegura que hay una predicción de emoción."
    
    # Crear el prompt
    prompt = f"""Escribe un poema corto de 4 líneas que conecte estas tres imágenes con la emocion de manera creativa:
    Imagen 1: {desc1}
    Imagen 2: {desc2}
    Imagen 3: {desc3}
    Emocion: {emotion_prediction}
    
    El poema debe ser en español, rimado y coherente, usando elementos de las descripciones y la emocion."""
    
    # Llamada a la API de OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Eres un poeta experto en español."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message["content"]

# Función para mostrar el prompt generado
def show_prompt(desc1, desc2, desc3):
    global emotion_prediction
    if not all([desc1, desc2, desc3]) or not emotion_prediction:
        return "Por favor, asegúrate de tener descripciones y una predicción de emoción antes de mostrar el prompt."
    prompt = f"""Escribe un poema corto de 4 líneas que conecte estas tres imágenes con la emocion de manera creativa:
    Imagen 1: {desc1}
    Imagen 2: {desc2}
    Imagen 3: {desc3}
    Emocion: {emotion_prediction}
    
    El poema debe ser en español, rimado y coherente, usando elementos de las descripciones y la emocion."""
    return prompt

# Función para resetear los cuadros de texto
def reset():
    return ""

# Interfaz de Gradio
with gr.Blocks() as demo:

    # Sección superior: Input de imagen y visualización
    gr.Markdown("### Subir y visualizar imagen")
    with gr.Row(equal_height=True):
        with gr.Column():
            image_input = gr.Image(label="Sube tu imagen aquí", type="pil", interactive=True)
        with gr.Column():
            image_output = gr.Image(label="Vista previa de la imagen", interactive=False)
            result_output = gr.Markdown("Predicción: ---")  # Lugar para mostrar la predicción
    reset_button_1 = gr.Button("Resetear Imagen", variant="secondary")

    # Lógica para enviar la imagen y actualizar la vista previa y el resultado
    image_input.change(
        send_image,
        inputs=image_input,
        outputs=[image_output, result_output]
    )

    # Lógica para resetear la sección superior
    reset_button_1.click(
        reset_image,
        outputs=[image_input, result_output]
    )

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
        poem_box = gr.Textbox(label="Poema generado", interactive=False)
        prompt_box = gr.Textbox(label="Prompt generado", interactive=False)  # Para mostrar el prompt

    with gr.Row():
        generate_poem_button = gr.Button("Generar Poema")
        show_prompt_button = gr.Button("Mostrar Prompt")  # Nuevo botón

        generate_poem_button.click(
            fn=generate_poem,
            inputs=descriptions,  # Usar descripciones generadas como entradas
            outputs=[poem_box]
        )
        show_prompt_button.click(
            fn=show_prompt,
            inputs=descriptions,  # Usa las descripciones generadas
            outputs=[prompt_box]
        )

    demo.launch()
