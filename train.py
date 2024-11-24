import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Rutas de los datasets
BASE_DIR = "Expression-1/"
TRAIN_CSV = os.path.join(BASE_DIR, "train/_classes.csv")
VALID_CSV = os.path.join(BASE_DIR, "valid/_classes.csv")
TEST_CSV = os.path.join(BASE_DIR, "test/_classes.csv")

# Resolución de entrada para el modelo
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Tamaño de batch y épocas de entrenamiento
BATCH_SIZE = 64
EPOCHS = 5

# Crear un directorio para guardar los resultados
def create_run_directory(base_dir="./runs"):
    """
    Crea un directorio numerado automáticamente para guardar modelos y resultados.
    
    Args:
        base_dir (str): Ruta base donde se crearán las carpetas de ejecución (por defecto: './runs').

    Returns:
        str: Ruta completa del nuevo directorio creado.
    """
    # Crear el directorio base si no existe
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Buscar el número de ejecución más alto
    existing_runs = [d for d in os.listdir(base_dir) if d.startswith("run_") and os.path.isdir(os.path.join(base_dir, d))]
    if existing_runs:
        # Extraer el número de cada carpeta
        run_numbers = [int(d.split("_")[1]) for d in existing_runs if d.split("_")[1].isdigit()]
        next_run = max(run_numbers) + 1
    else:
        next_run = 1

    # Crear la nueva carpeta
    new_run_dir = os.path.join(base_dir, f"run_{next_run}")
    os.makedirs(new_run_dir)

    return new_run_dir


# Función para cargar y preparar DataFrame
def load_and_prepare_dataframe(csv_path, base_dir):
    df = pd.read_csv(csv_path)
    # Limpiar nombres de columnas
    df.columns = df.columns.str.strip()  # Eliminar espacios en los nombres de columnas
    # Agregar la ruta completa a cada archivo de imagen
    df['filename'] = df['filename'].apply(lambda x: os.path.join(base_dir, x))
    return df


# Crear carpeta para guardar resultados
run_dir = create_run_directory()

# Definir rutas específicas para guardar el modelo y otros resultados
model_path = os.path.join(run_dir, "best_model.keras")
final_model_path = os.path.join(run_dir, "multi_label_model.keras")
final_model_path_h5 = os.path.join(run_dir, "multi_label_model.h5")
history_plot_path = os.path.join(run_dir, "training_history.png")

# Cargar los DataFrames
train_df = load_and_prepare_dataframe(TRAIN_CSV, os.path.join(BASE_DIR, "train"))
valid_df = load_and_prepare_dataframe(VALID_CSV, os.path.join(BASE_DIR, "valid"))
test_df = load_and_prepare_dataframe(TEST_CSV, os.path.join(BASE_DIR, "test"))

# Definir los nombres de las etiquetas sin espacios
label_columns = ['Asustado', 'Desagradable', 'Enojado', 'Feliz', 'Neutral', 'Sorprendido', 'Triste']

# Verificar y rellenar valores nulos
for df in [train_df, valid_df, test_df]:
    df[label_columns] = df[label_columns].fillna(0)

# **Definición de ImageDataGenerator**
train_datagen = ImageDataGenerator(rescale=1.0/255)  # Escala los valores de píxeles a [0, 1]
valid_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

# **Generadores de datos**
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col=label_columns,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='raw',  # Para multi-label
    shuffle=True
)

valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=valid_df,
    x_col='filename',
    y_col=label_columns,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='raw',
    shuffle=False
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col=label_columns,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='raw',
    shuffle=False
)

# Modelo base preentrenado
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Congelar capas preentrenadas
for layer in base_model.layers:
    layer.trainable = False

# Construir el modelo
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(7, activation='sigmoid')(x)  # 7 etiquetas, activación sigmoid para multi-label

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss"),
    EarlyStopping(patience=10, monitor="val_loss", restore_best_weights=True)
]



# Entrenamiento del modelo
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=valid_generator.samples // BATCH_SIZE,
    callbacks=callbacks
)


# Evaluar el modelo con datos de prueba
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Guardar el modelo entrenado
model.save(final_model_path)
#model.save(final_model_path_h5)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print(f'\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

# Mostrar la gráfica de accuracy y pérdida
plt.figure(figsize=(14, 6))

# Subgráfica de Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy durante el entrenamiento')
plt.ylabel('Accuracy')
plt.xlabel('Épocas')
plt.legend(['Train', 'Validation'], loc='upper left')

# Subgráfica de Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss durante el entrenamiento')
plt.ylabel('Loss')
plt.xlabel('Épocas')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# Obtener las etiquetas verdaderas y predichas
y_true = []
y_pred = []

for i in range(len(test_generator)):
    images, labels = test_generator[i]  # Obtener imágenes y etiquetas reales del batch
    predictions = model.predict(images)  # Obtener predicciones
    y_true.extend(labels)  # Etiquetas reales
    y_pred.extend((predictions > 0.01).astype(int))  # Umbral para etiquetas multi-label

# Convertir listas a arrays
y_true = np.array(y_true).reshape(-1, len(label_columns))
y_pred = np.array(y_pred).reshape(-1, len(label_columns))

# Calcular métricas
conf_mats = []
reports = []

print("\nReporte de clasificación por clase:")
for i, label in enumerate(label_columns):
    # Calcular matriz de confusión para cada etiqueta
    cm = confusion_matrix(y_true[:, i], y_pred[:, i])
    conf_mats.append(cm)

    # Reporte de clasificación para cada clase
    report = classification_report(y_true[:, i], y_pred[:, i], target_names=[f'No {label}', label])
    reports.append(report)
    print(f"\nEtiqueta: {label}\n{report}")

# Graficar accuracy por clase
accuracy_per_class = []
for i, label in enumerate(label_columns):
    report_dict = classification_report(y_true[:, i], y_pred[:, i], target_names=[f'No {label}', label], output_dict=True)
    accuracy_per_class.append(report_dict[label]['precision'])

plt.figure(figsize=(10, 6))
sns.barplot(x=label_columns, y=accuracy_per_class, palette="Blues")
plt.title("Precisión por clase")
plt.xlabel("Clases")
plt.ylabel("Precisión (Accuracy)")
plt.xticks(rotation=45)
plt.savefig(os.path.join(run_dir, "accuracy_per_class.png"))
