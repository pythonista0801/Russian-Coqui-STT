# Russian-Coqui-STT

Guidelines for training Russian Coqui STT model

There used to be two most popular STT workhorses, Coqui STT and DeepSpeech. However, the developers have been told to abandon the DeepSpeech model and they focused all their efforts entirely on Coqui. The code for manually installing and training the Coqui STT model is provided below. The complete instructions are available at Coqui STT website. Prerequisites for the installation must be the following: 
- Python 3.6, 3.7 or 3.8, 
- Mac or Linux environment, 
- CUDA 10.0 and CuDNN v7.6

You may adhere to the following guidelines to train STT model from Linux terminal window in a few simple steps as well as use a notebook available in this repository for audio data preprocessing and formatting in accordance with Coqui STT instruction.

1. Download. Clone the STT repository from GitHub
```
$ git clone https://github.com/coqui-ai/STT
```

2. Installing STT and its dependencies is better with a virtual environment. Python’s built-in venv module is recommended to manage a Python environment. Let’s setup a Python virtual environment, and name it coqui-stt-train-venv (as advised in the original setup instructions):
```
$ python3 -m venv coqui-stt-train-venv
```
Activate the virtual environment.
```
$ source coqui-stt-train-venv/bin/activate
```
Install dependencies and STT. Now that we have cloned the STT repo from Github and setup a virtual environment with venv, we can install STT and its dependencies. Python’s built-in pip module is recommended for installation:
```
$ cd ./STT
$ python -m pip install --upgrade pip wheel setuptools
$ python -m pip install --upgrade -e .
```

3. GPU support. If you have an NVIDIA GPU, it is highly recommended to install TensorFlow with GPU support. Training will be significantly faster than using the CPU.
```
$ python -m pip uninstall tensorflow
$ python -m pip install 'tensorflow-gpu==1.15.4'
```

4. Verify the installation.
```
$ ./bin/run-ldc93s1.sh
```

5. Training Data. There’s two kinds of data needed to train an STT model: audio clips and text transcripts. Data format. “Audio data is expected to be stored as WAV, sampled at 16kHz, and mono-channel. There’s no hard expectations for the length of individual audio files, but in our experience, training is most successful when WAV files range from 5 to 20 seconds in length.” It is best to preprocess transcripts and remove capital letters and punctuation to make it easier for the model to process the text. The dataset, known as common voice, can be downloaded from the hugging face transformers library or directly from the mozilla website. My audio preprocessing file is available in this repository as AudioProcessing.ipynb.

CSV file format. The audio and transcripts used in training are specified via CSV files. You should supply CSV files for training (train.csv), validation (dev.csv), and testing (test.csv). The CSV files should contain three columns:
- wav_filename - the path to a WAV file on your machine
- wav_filesize - the number of bytes in the WAV file
- transcript - the text transcript of the WAV file

6. Augmenting audio data. The data can be augmented with one of the libraries discussed in the notebook or directly with Coqui tools. 
The bin/data_set_tool.py tool supports --augment parameters (for sample domain augmentations) and can be used for experimenting with different configurations or creating augmented data sets. However, not all of the specified augmentations can get applied, as this tool only supports overlay, codec, reverb, resample and volume. Let’s see how it works.
```
$ ./bin/data_set_tool.py /
--augment "overlay[p=0.5,source=noise.sdb,layers=1,snr=50:20~10]" /
--augment "reverb[p=0.1,delay=50.0~30.0,decay=10.0:2.0~1.0]"/
--augment "resample[p=0.1,rate=12000:8000~4000]" /
--augment "codec[p=0.1,bitrate=48000:16000]" /
--augment "volume[p=0.1,dbfs=-10:-40]" / 
--sources train.csv /
--target train-augmented.csv
```

7. Bootstrap from a pre-trained model. When the STT model is installed and the data is prepared, you may start running the training process. The checkpoints for the Russian pre-trained release model are available here: https://coqui-ai-public-data.s3.amazonaws.com/russian_checkpoints_coqui_v010_joe_meyer.zip. Therefore, there will be no need to start training the model from scratch. Instead, you can apply bootstrapping technique and fine-tune the existing model on your data. 

“There are currently two supported approaches to bootstrapping from a pre-trained 🐸STT model: fine-tuning or transfer-learning. Choosing which one to use depends on your target dataset. Does your data use the same alphabet as the release model? If “Yes”, then you fine-tune. If “No”, then you use transfer-learning.”

“Currently, 🐸STT release models are trained with --n_hidden 2048, so you need to use that same value when initializing from the release models. Release models are also trained with --train_cudnn, so you’ll need to specify that as well. If you don’t have a CUDA compatible GPU, then you can workaround it by using the --load_cudnn true.”

Fine-tuning (same alphabet):
```
$ python -m coqui_stt_training.train \
--n_hidden 2048
--checkpoint_dir ./checkpoints_joe_meyer \ 
--epochs 30
--learning_rate 0.0001
--dropout_rate 0.3
--train_files train-augmented.csv \
--dev_files dev-augmented.csv \
--test_files test-augmented.csv \
--load_cudnn true
```

Transfer learning (different alphabet):
```
python -m coqui_stt_training.train \
    --drop_source_layers 1 \
    --alphabet_config_path my-alphabet.txt \
    --save_checkpoint_dir path/to/output-checkpoint/folder \
    --load_checkpoint_dir path/to/input-checkpoint/folder \
    --train_files my-new-language-train.csv \
    --dev_files   my-new-language-dev.csv \
    --test_files  my-new-language-test.csv
```

8. Deployment…
