from roboflow import Roboflow
import os
import cv2

# Ruta de la imagen
image_file = "img/img/WhatsApp Image 2024-11-24 at 2.10.37 PM (1).jpeg"
image = cv2.imread(image_file)

if image is None:
    raise ValueError(f"Error al cargar la imagen: {image_file}")

# Carga la API key desde las variables de entorno
api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise ValueError("La clave de API de Roboflow no está configurada. Usa 'export ROBOFLOW_API_KEY=<your api key>' para configurarla.")

# Configuración del proyecto y modelo
rf = Roboflow(api_key=api_key)
project = rf.workspace().project("expression-bivfq-pugqb")  # Asegúrate de que el endpoint sea correcto
model = project.version(1).model  # Cambia el número de versión si es necesario

# Realizar predicción en la imagen local
results = model.predict(image_file).json()
print("Resultados de predicción:", results)

# Extraer las predicciones de la lista
all_predictions = results.get("predictions", [])
if not all_predictions:
    raise ValueError("No se encontraron predicciones en la respuesta del modelo.")

# Asumimos que estamos interesados en la primera predicción
first_prediction = all_predictions[0]
class_predictions = first_prediction.get("predictions", {})

# Visualizar las predicciones en la imagen
font = cv2.FONT_HERSHEY_SIMPLEX
y_offset = 50
for label, data in class_predictions.items():
    confidence = data.get("confidence", 0)
    text = f"{label}: {confidence:.2f}"
    cv2.putText(image, text, (10, y_offset), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    y_offset += 30

# Mostrar la imagen con las predicciones
cv2.imshow("Predicciones", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
