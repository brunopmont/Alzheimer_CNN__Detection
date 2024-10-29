import numpy as np
import nibabel as nib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# Função para carregar imagens NIfTI, seus rótulos e cortar as imagens
def load_nifti_images_with_labels(base_dir, slice_idx):
    images = []
    labels = []
    
    # Caminhos das subpastas
    for label in ['cn', 'mci', 'ad']:
        label_dir = os.path.join(base_dir, label)
        for fname in os.listdir(label_dir):
            img_path = os.path.join(label_dir, fname)
            img = nib.load(img_path).get_fdata()
            img_cropped = img[slice_idx[0], slice_idx[1], slice_idx[2]]
            images.append(img_cropped)
            labels.append(label)

    # Codificando os rótulos
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    return np.array(images), labels_encoded, label_encoder.classes_

# Função para criar o modelo
def create_model(input_shape):
    model = Sequential([
        Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling3D(pool_size=(2, 2, 2)),
        BatchNormalization(),
        
        Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
        MaxPooling3D(pool_size=(2, 2, 2)),
        BatchNormalization(),
        
        Conv3D(128, (3, 3, 3), activation='relu', padding='same'),
        MaxPooling3D(pool_size=(2, 2, 2)),
        BatchNormalization(),
        
        Conv3D(256, (3, 3, 3), activation='relu', padding='same'),
        MaxPooling3D(pool_size=(2, 2, 2)),
        BatchNormalization(),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')  # 3 classes: CN, MCI, AD
    ])
    return model

# Definindo caminhos
base_dir = os.getenv('BASE_DIR', '/default/base/path')
train_dir = os.path.join(base_dir, 'ADNI1_ADNI2', 'train')
val_dir = os.path.join(base_dir, 'ADNI1_ADNI2', 'validation')
results_dir = os.path.join(base_dir, 'ADNI1_ADNI2', 'results')

# Índices de corte para o corte NIfTI
SLICE_NII_IDX0 = slice(24, 169)
SLICE_NII_IDX1 = slice(24, 206)
SLICE_NII_IDX2 = slice(6, 161)

# Criar o diretório de resultados se ele não existir
os.makedirs(results_dir, exist_ok=True)

# Carregar os dados com cortes
train_images, train_labels, class_labels = load_nifti_images_with_labels(train_dir, 
                                                                        (SLICE_NII_IDX0, SLICE_NII_IDX1, SLICE_NII_IDX2))
val_images, val_labels, _ = load_nifti_images_with_labels(val_dir, 
                                                           (SLICE_NII_IDX0, SLICE_NII_IDX1, SLICE_NII_IDX2))

# Adicionar a dimensão do canal
train_images = train_images.reshape((-1, *train_images.shape[1:], 1))
val_images = val_images.reshape((-1, *val_images.shape[1:], 1))

# Criar e compilar o modelo
input_shape = (145, 182, 155, 1)  # Formato para dados em escala de cinza
model = create_model(input_shape)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])

# Callbacks de treinamento
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

# Treinamento do modelo
history = model.fit(
    train_images, 
    train_labels,
    epochs=100,
    verbose=1,
    validation_data=(val_images, val_labels),
    callbacks=[early_stopping, reduce_lr],
    batch_size=32
)

# Salva o modelo
model.save(os.path.join(results_dir, "trained_model.h5"))

# Função para avaliação do modelo
def evaluate_model(val_images, val_labels, model, results_dir):
    predictions = model.predict(val_images)
    predicted_classes = np.argmax(predictions, axis=1)

    report = classification_report(val_labels, predicted_classes, target_names=class_labels, output_dict=True)
    
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(classification_report(val_labels, predicted_classes, target_names=class_labels))
    print("Classification Report:\n", classification_report(val_labels, predicted_classes, target_names=class_labels))
    
    conf_matrix = confusion_matrix(val_labels, predicted_classes)
    np.savetxt(os.path.join(results_dir, "confusion_matrix.txt"), conf_matrix, fmt='%d')
    
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()

    # Exibindo as métricas
    accuracy = report['accuracy']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']

    print(f"Acurácia: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

evaluate_model(val_images, val_labels, model, results_dir)

# Função para plotar e salvar o histórico de treinamento
def plot_history(history, results_dir):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.title('Acurácia ao Longo das Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Perda de Treinamento')
    plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.title('Perda ao Longo das Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_history.png"))
    plt.close()

plot_history(history, results_dir)