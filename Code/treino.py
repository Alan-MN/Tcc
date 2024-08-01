import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Reshape, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Função para calcular MFCC
def compute_mfcc(audio_file, n_mfcc=13, n_fft=2048, hop_length=512):
    audio, sr = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfcc, sr

def list_wav_files(root_dir):
    wav_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.wav'):
                wav_files.append(os.path.join(dirpath, filename))
    return wav_files

# Funções para plotar MFCC
def plot_mfcc(mfcc, sr, hop_length, title='MFCC'):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, sr=sr, hop_length=hop_length, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_mfcc_side_by_side(mfcc1, sr1, mfcc2, sr2, hop_length, title1='MFCC 1', title2='MFCC 2'):
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))

    # Plotar primeiro MFCC
    axs[0].set_title(title1)
    librosa.display.specshow(mfcc1, sr=sr1, hop_length=hop_length, x_axis='time', ax=axs[0])
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('MFCC Coefficients')

    # Plotar segundo MFCC
    axs[1].set_title(title2)
    librosa.display.specshow(mfcc2, sr=sr2, hop_length=hop_length, x_axis='time', ax=axs[1])
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('MFCC Coefficients')

    # Ajustar layout
    plt.tight_layout()
    plt.show()    

# Função para extrair características e rótulos
def extract_features(input_dir):
    features = []
    labels = []
    real_files = []
    fake_files = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                mfcc, sr = compute_mfcc(file_path)
                features.append(mfcc)
                
                # Determinar o rótulo com base no nome do diretório
                if 'fake' in root:
                    labels.append(1)
                    fake_files.append(file_path)
                else:
                    labels.append(0)
                    real_files.append(file_path)
                    
    return np.array(features), np.array(labels), real_files, fake_files

#Funçâo para treinar Multi Layer Perceptron
def train_mlp(X_train, y_train, batch_size):
    X_train_flattened = X_train.reshape(X_train.shape[0], -1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flattened)
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=1e-4, solver='adam', verbose=10, random_state=42, learning_rate_init=.001, batch_size=batch_size)
    mlp.fit(X_train_scaled, y_train)
    return mlp, scaler

# Função para treinar Random Forest
def train_random_forest(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    return clf

# Função para treinar LSTM
def train_lstm(X_train, y_train, input_shape, batch_size, epochs):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid', dtype='float32'))
    
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

# Função para treinar SCNN
def train_shallow_cnn(X_train, y_train, input_shape, batch_size, epochs):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', dtype='float32'))
    
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

# Função para avaliar o modelo
def evaluate_model(model, X_test, y_test, model_type, scaler=None):
    if model_type in ['SimpleLSTM', 'ShallowCNN']:
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
    else:
        if scaler is not None:
            X_test_flattened = X_test.reshape(X_test.shape[0], -1)
            X_test_scaled = scaler.transform(X_test_flattened)
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)
        
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    
    print(f'Evaluation results for {model_type}:')
    print(f'Accuracy: {accuracy}')
    print(f'F1 Score: {f1}')
    print(f'Recall: {recall}')
    print()

# Função para salvar características e rótulos
def save_features_labels(features, labels, feature_save_path, labels_save_path):
    np.save(feature_save_path, features)
    np.save(labels_save_path, labels)

# Função para carregar características e rótulos salvos
def load_features_labels(feature_save_path, labels_save_path):
    features = np.load(feature_save_path)
    labels = np.load(labels_save_path)
    return features, labels

# Diretório de entrada
input_directory = r'C:\path_to_origem\Cleaned'
feature_save_path = r'C:\path_to_origem\Extracted\mfcc_features.npy'
labels_save_path = r'C:\path_to_origem\Extracted\labels.npy'

# Lista de arquivos específicos para plotar
if os.path.exists(feature_save_path) and os.path.exists(labels_save_path):
    print("Loading saved features and labels...")
    features, labels = load_features_labels(feature_save_path, labels_save_path)
    
    # Selecionar aleatoriamente um arquivo real e um arquivo falso para plotar
    fake_files = list_wav_files(os.path.join(input_directory, 'fake'))
    real_files = list_wav_files(os.path.join(input_directory, 'real'))
    
    if len(fake_files) > 0 and len(real_files) > 0:
        random_fake_file = np.random.choice(fake_files)
        random_real_file = np.random.choice(real_files)
        mfcc_fake, sr_fake = compute_mfcc(random_fake_file)
        mfcc_real, sr_real = compute_mfcc(random_real_file)
        
        if mfcc_fake is not None and sr_fake is not None and mfcc_real is not None and sr_real is not None:
            plot_mfcc_side_by_side(mfcc_fake, sr_fake, mfcc_real, sr_real, hop_length=512, title1=f'MFCC for {random_fake_file}', title2=f'MFCC for {random_real_file}')
    else:
        print("No .wav files found in the 'fake' directory or its subdirectories.")
else:
    print("Extraindo características")
    features, labels, real_files, fake_files = extract_features(input_directory)
    save_features_labels(features, labels, feature_save_path, labels_save_path)
    print("Características extraídas e salvas")
    
    # Selecionar aleatoriamente um arquivo real e um arquivo falso para plotar
    fake_files = list_wav_files(os.path.join(input_directory, 'fake'))
    real_files = list_wav_files(os.path.join(input_directory, 'real'))
    
    # Calcular MFCCs para os arquivos selecionados
    if len(fake_files) > 0:
        random_fake_file = np.random.choice(fake_files)
        mfcc_fake, sr_fake = compute_mfcc(random_fake_file)
        if mfcc_fake is not None and sr_fake is not None:
            plot_mfcc(mfcc_fake, sr_fake, hop_length=512, title=f'MFCC for {random_fake_file}')
    else:
        print("No .wav files found in the 'fake' directory or its subdirectories.")

# Verificar as dimensões das características
print(f'Dimensão das características: {features.shape}')
batch_size = 32
epochs = 10

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("Time =", dt_string)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Treinar e avaliar MLP
print("Training MLP...")
mlp_model, scaler = train_mlp(X_train, y_train, batch_size)
print("MLP trained.")
evaluate_model(mlp_model, X_test, y_test, "MLP", scaler)

# Treinar e avaliar LSTM
print("Training Simple LSTM...")
input_shape_lstm = (features.shape[1], features.shape[2])
lstm_model = train_lstm(X_train, y_train, input_shape_lstm, batch_size, epochs)
print("Simple LSTM trained.")
evaluate_model(lstm_model, X_test, y_test, "Simple LSTM")

# Treinar e avaliar SCNN
print("Training Shallow CNN...")
input_shape_cnn = (features.shape[1], features.shape[2], 1)
X_train_cnn = np.expand_dims(X_train, -1)
X_test_cnn = np.expand_dims(X_test, -1)
scnn_model = train_shallow_cnn(X_train_cnn, y_train, input_shape_cnn, batch_size, epochs)
print("Shallow CNN trained.")
evaluate_model(scnn_model, X_test_cnn, y_test, "Shallow CNN")

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("Time =", dt_string)
