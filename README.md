# RnD-Project
| DESCRIPTION |
|-------------|
| This is a step-by-step guide to setting up the proper virtual environment, with which I worked during the project. The steps denote whether the following line of text is a CLI command by marking it with (CLI), or whether it is textual instructions, denoted by the lack of (CLI). |


##--------------------STEP 0: Preparations for VENV--------------------##<br>
Download Anaconda3 from https://www.anaconda.com/download and install the software
- (CLI) module load GCC/12.3.0
- (CLI) module avail CUDA

##--------------------STEP 1: Setting up the VENV--------------------##<br>
Install Anaconda 3 in /home/user/
- (CLI) bash Anaconda3-2025.06-0-Linux-x86_64.sh
- (CLI) conda init
- (CLI) conda config --set auto_activate_base False
- (CLI) conda activate
- (CLI) mamba create -n nlp-gpu python=3.10
- (CLI) conda activate nlp-gpu
- (CLI) pip install torch=2.3.1 --index-url https://download.pytorch.org/whl/cu121
- (CLI) pip install torch torchvision torchaudio accelerate nltk opennmt=1.2.0 sentencepiece ctranslate2 huggingface_hub nbformat pandas sentence_transformers spacy spacy-curated-transformers
- (CLI) conda install -c nvidia cuda-toolkit=11.8

##--------------------STEP 2: Preparing data--------------------##<br>
Retrieve the data 'paraphrases.zip' [NOTE: The data retrieved from the WMT-19 dataset]
- (CLI) unzip paraphrases.zip

##--------------------STEP 3: Preparing translation models--------------------##<br>
[Facit for NMT: https://github.com/ymoslem/Adaptive-MT-LLM/blob/main/MT/NLLB.ipynb]<br>
[Facit for LLM: https://huggingface.co/meta-llama/Llama-2-13b-chat-hf/tree/main?library=transformers]

For the LLM:<br>
Request access to Meta's models on HuggingFace<br>
Generate access token<br>
- (CLI) hf auth login
Paste [HuggingFace access token]

For the LLM:
Request access to Meta's models on HuggingFace
Generate access token
- (CLI) hf auth login
Paste [HuggingFace access token]
