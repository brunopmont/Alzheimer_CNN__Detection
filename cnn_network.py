import numpy as np
import nibabel
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Função para criar o modelo
def create_model(input_shape):
    model = Sequential()
    model.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization())
    
    model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization())
    
    model.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization())
    
    model.add(Conv3D(256, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))  # 3 classes: CN, MCI, AD

    return model

# Definindo os diretórios
train_dir = "C:\\Users\\bruno\\Desktop\\IANS\\ADNI1_ADNI2\\train"
val_dir = "C:\\Users\\bruno\\Desktop\\IANS\\ADNI1_ADNI2\\validation"

# Criando geradores de dados
train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

# Carregando dados usando flow_from_directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(145, 182, 155),  # Ajuste de acordo com suas imagens
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=8,
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(145, 182, 155),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=8,
    shuffle=False
)

# Criar e compilar o modelo
input_shape = (145, 182, 155, 1)  # Para dados em escala de cinza
model = create_model(input_shape)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Criar o callback de EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Ou 'val_accuracy', dependendo de sua preferência
    patience=10,         # Número de épocas sem melhoria antes de parar
    verbose=1,           # Imprimir mensagens de progresso
    restore_best_weights=True  # Restaura os melhores pesos do modelo
)

# Treinamento do modelo com EarlyStopping e armazenando a história
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[early_stopping]
)

# Função para calcular e exibir as métricas
def evaluate_model(generator):
    # Prever as classes usando o gerador
    predictions = model.predict(generator)
    predicted_classes = np.argmax(predictions, axis=1)

    # Obter as classes verdadeiras
    true_classes = generator.classes
    class_labels = list(generator.class_indices.keys())

    # Calcular e exibir as métricas
    print("Classification Report:\n", classification_report(true_classes, predicted_classes, target_names=class_labels))
    print("Confusion Matrix:\n", confusion_matrix(true_classes, predicted_classes))

# Avaliar o modelo com os dados de validação
evaluate_model(val_generator)

# Plotando os gráficos de acurácia e perda
def plot_history(history):
    # Plotando a acurácia
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.title('Acurácia ao Longo das Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()

    # Plotando a perda
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Perda de Treinamento')
    plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.title('Perda ao Longo das Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Chamar a função para plotar
plot_history(history)
