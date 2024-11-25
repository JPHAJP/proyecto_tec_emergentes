import gradio as gr

# Función para resetear la imagen del input
def reset_image():
    return None

# Interfaz principal
with gr.Blocks(css=".main-title { font-size: 24px; font-weight: bold; text-align: center; margin-bottom: 20px; }") as demo:
    # Título principal
    gr.Markdown("<div class='main-title'>Cargador y Reseteador de Imágenes</div>")

    # Sección superior: Input de imagen y visualización
    gr.Markdown("### Subir y visualizar imagen")
    with gr.Row(equal_height=True):
        with gr.Column():
            image_input = gr.Image(label="Sube tu imagen aquí", type="pil", interactive=True)
        with gr.Column():
            image_output = gr.Image(label="Vista previa de la imagen", interactive=False)
    reset_button_1 = gr.Button("Resetear Imagen", variant="secondary")

    # Lógica para actualizar la vista previa
    image_input.change(lambda img: img, inputs=image_input, outputs=image_output)

    # Lógica para resetear la sección superior
    reset_button_1.click(reset_image, outputs=image_input)
    reset_button_1.click(reset_image, outputs=image_output)

    # Sección inferior: 3 inputs de imágenes
    gr.Markdown("### Subir varias imágenes")
    with gr.Row(equal_height=True):
        image_input_1 = gr.Image(label="Imagen 1", type="pil", interactive=True)
        image_input_2 = gr.Image(label="Imagen 2", type="pil", interactive=True)
        image_input_3 = gr.Image(label="Imagen 3", type="pil", interactive=True)
    reset_button_2 = gr.Button("Resetear Imágenes", variant="secondary")

    # Lógica para resetear las imágenes de la sección inferior
    reset_button_2.click(reset_image, outputs=[image_input_1, image_input_2, image_input_3])

# Ejecutar la interfaz
demo.launch()
