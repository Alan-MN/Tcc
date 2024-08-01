import os
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import detect_silence

#Funcionalidades:
#Remoção do silencio
#Padronização de duração
#Padronização de frequencia
#Duração da execução--- 3:30h media

def load_audio(file_path, sample_rate=16000):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio, sr

def remove_silence(audio, sample_rate, silence_thresh=-40.0, min_silence_len=200):
    # Convert audio para o pydub AudioSegment
    audio_segment = AudioSegment(
        audio.tobytes(), 
        frame_rate=sample_rate,
        sample_width=audio.dtype.itemsize, 
        channels=1
    )
    
    # Detecta segmentos silenciosos
    silent_ranges = detect_silence(
        audio_segment, 
        min_silence_len=min_silence_len, 
        silence_thresh=silence_thresh
    )
    
    non_silent_ranges = []
    prev_end = 0
    for start, end in silent_ranges:
        if start > prev_end:
            non_silent_ranges.append((prev_end, start))
        prev_end = end
    if prev_end < len(audio_segment):
        non_silent_ranges.append((prev_end, len(audio_segment)))
    
    # Concatena frequncias nao silenciosas
    non_silent_audio = np.concatenate([
        audio[int(start / 1000 * sample_rate):int(end / 1000 * sample_rate)]
        for start, end in non_silent_ranges
    ])
    
    return non_silent_audio

def standardize_duration(audio, sample_rate, target_duration):
    current_duration = len(audio) / sample_rate
    if current_duration > target_duration:
        audio = audio[:int(target_duration * sample_rate)]
    else:
        padding = int((target_duration - current_duration) * sample_rate)
        audio = np.pad(audio, (0, padding), 'constant')
    return audio

def process_audio_file(file_path, output_path, sample_rate=16000, target_duration=5.0):
    audio, sr = load_audio(file_path, sample_rate)
    audio = remove_silence(audio, sr, silence_thresh=-40.0, min_silence_len=200)
    audio = standardize_duration(audio, sr, target_duration)
    # Salvar o áudio processado usando soundfile
    sf.write(output_path, audio, sample_rate)

def process_dataset(input_dir, output_dir, sample_rate=16000, target_duration=5.0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                output_dir_for_file = os.path.dirname(output_path)
                
                if not os.path.exists(output_dir_for_file):
                    os.makedirs(output_dir_for_file)
                
                process_audio_file(input_path, output_path, sample_rate, target_duration)

# Exemplo de uso
input_directory = r'C:\path_to_origem'
output_directory = r'C:\path_to_code\Cleaned\(real ou fake)' #Por enquanto deve ser feito um de cada vez, limpeza dos datasets reais e falsos
process_dataset(input_directory, output_directory)
