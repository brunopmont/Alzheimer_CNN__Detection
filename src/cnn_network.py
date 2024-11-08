import numpy as np
import nibabel as nib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import sys

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
        Input(shape=input_shape),

        # Camada 1
        Conv3D(32, (3, 3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding='same'),

        # Camada 2
        Conv3D(64, (3, 3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding='same'),

        # Camada 3
        Conv3D(128, (3, 3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv3D(128, (3, 3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding='same'),

        # Camada 4
        Conv3D(256, (3, 3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv3D(256, (3, 3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding='same'),

        # Camada 5
        Conv3D(256, (3, 3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv3D(256, (3, 3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding='same'),

        # Flatten e camadas densas
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        # Camada de saída
        Dense(3, activation='softmax')  # 3 classes: CN, MCI, AD
    ])
    return model

# Função para avaliação
def evaluate_model(val_images, val_labels, model, results_dir):
    predictions = model.predict(val_images)
    predicted_classes = np.argmax(predictions, axis=1)

    report = classification_report(val_labels, predicted_classes, target_names=class_labels, output_dict=True)
    
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(classification_report(val_labels, predicted_classes, target_names=class_labels))
    print("Classification Report:\n", classification_report(val_labels, predicted_classes, target_names=class_labels))
    
    conf_matrix = confusion_matrix(val_labels, predicted_classes)
    
    # Normalizar a matriz de confusão para porcentagens
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    np.savetxt(os.path.join(results_dir, "confusion_matrix.txt"), conf_matrix, fmt='%d')
    
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)
    
    # Adicionando os rótulos com porcentagens
    thresh = conf_matrix_normalized.max() / 2.
    for i, j in np.ndindex(conf_matrix_normalized.shape):
        plt.text(j, i, f'{conf_matrix_normalized[i, j]:.2%}', 
                 horizontalalignment="center",
                 color="white" if conf_matrix_normalized[i, j] > thresh else "black")
    
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

# Salvar hiperparâmetros
def save_hyperparameters(model, results_dir):
    n = len(os.listdir(results_dir))
    folder_name = "test_" + str(n+1)
    results_dir = os.path.join(results_dir, folder_name)

    with open(os.path.join(results_dir, "model_hyperparameters.txt"), "w") as f:
            f.write(model.get_config())

base_path = ''

# Caminho base
if len(sys.argv > 1):
    base_path = sys.argv[1]
else:
    base_path = os.getcwd

# Definindo caminhos
train_dir = os.path.join(base_path, 'train')
val_dir = os.path.join(base_path, 'validation')
results_dir = os.path.join(base_path, 'results')

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

# Embaralhar os dados
train_images, train_labels = shuffle(train_images, train_labels, random_state=42)
val_images, val_labels = shuffle(val_images, val_labels, random_state=42)

# Taxa de aprendizado
learning_rate = 1e-5
optimizer = Adam(learning_rate=learning_rate)

# Compila model
input_shape = (145, 182, 155, 1)
model = create_model(input_shape)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

# Treinamento
history = model.fit(
    train_images, 
    train_labels,
    epochs=50,
    verbose=1,
    validation_data=(val_images, val_labels),
    callbacks=[early_stopping, reduce_lr],
    batch_size=16
)

# Salva o modelo
model.save(os.path.join(results_dir, "trained_model.h5"))

save_hyperparameters(model)

evaluate_model(val_images, val_labels, model, results_dir)

plot_history(history, results_dir)