#*TCC*

Repositório destinado ao desenvolvimento do projeto de TCC de detecção de Deepfakes em áudio.

Para utilizar este projeto precisaremos nos atentar as dependencias compativeis com o seu ambiente.
Para começar, precisaremos do CUDA 11.2(https://developer.nvidia.com/cuda-11.2.0-download-archive) e do cuDNN 8.1.1(https://developer.nvidia.com/rdp/cudnn-archive) 
As demais dependencias podem ser instaladas a partir do comando

pip install -r requirements.txt

--Pre processamento do dataset

Para termos uma maior eficiencia na classificação, devemos primeiro ter certeza que os audios estão em um mesmo padrão. portanto iremos utilizar o preprocess.py para fazer isso.
Ele ira remover trechos ruidosos, padronizar a duração dos áudios e padronizar a frequencia dos áudios.
Dentro do arquivo altere input_directory e output_directory para a origem do seu dataset e o local idealmente devera ser a pasta real ou fake dentro de Code\Cleaned.

Evaluation results for Random Forest:
Accuracy: 0.9120241568840334
F1 Score: 0.9539881110815367
Recall: 0.999962800386876

Evaluation results for MLP:
Accuracy: 0.9120580850919454
F1 Score: 0.954006671871673
Recall: 1.0

Evaluation results for Simple LSTM:
Accuracy: 0.9120580850919454
F1 Score: 0.954006671871673
Recall: 1.0

Evaluation results for Shallow CNN:
Accuracy: 0.9120580850919454
F1 Score: 0.954006671871673
Recall: 1.0