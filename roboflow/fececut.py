from ultralytics import YOLO
import cv2

# Función para recortar la cara con mayor precisión
def crop_highest_accuracy_face(image, boxes, confidences):
    if boxes.size == 0:  # Comprobamos si el array está vacío
        return None  # No se detectaron caras
    max_index = confidences.index(max(confidences))  # Índice de la mayor confianza
    x1, y1, x2, y2 = map(int, boxes[max_index][:4])
    face = image[max(0, y1):y2, max(0, x1):x2]
    return face

# Main
def cut():
    # Ruta al modelo YOLOv8
    model_path = "gradio/yolov8m-face.pt"

    # Cargar el modelo
    model = YOLO(model_path)

    # Cargar la imagen
    image_path = "people.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: No se pudo cargar la imagen {image_path}")
        return

    # Realizar la detección
    results = model(image)

    # Obtener las bounding boxes y las confianzas
    detections = results[0].boxes.xyxy  # Coordenadas de las cajas [x1, y1, x2, y2]
    confidences = results[0].boxes.conf.cpu().numpy().tolist()  # Confianzas de las detecciones
    boxes = detections.cpu().numpy()   # Convertir a numpy para manejarlo fácilmente

    print(f"Se detectaron {len(boxes)} caras")

    if len(boxes) > 0:
        # Recortar la cara con mayor precisión
        face = crop_highest_accuracy_face(image, boxes, confidences)
        if face is not None:
            face_path = "just_face.jpg"
            cv2.imwrite(face_path, face)
            print(f"Guardada la cara con mayor precisión en: {face_path}")
        else:
            print("No se pudo recortar la cara con mayor precisión.")
    else:
        print("No se detectaron caras en la imagen.")

    # Mostrar la imagen original con la detección de mayor precisión
    if len(boxes) > 0:
        max_index = confidences.index(max(confidences))  # Índice de mayor confianza
        x1, y1, x2, y2 = map(int, boxes[max_index][:4])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"Conf: {confidences[max_index]:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # cv2.imshow("Detección de cara con mayor precisión", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    cut()
