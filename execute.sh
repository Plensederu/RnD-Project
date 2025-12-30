#!/bin/bash
#SBATCH -A uppmax2025-3-5
#SBATCH -M pelle
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --mail-type=END
#SBATCH -t 10:00:00

# load module & package
module load GCC/12.3.0

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

# setup conda environment and test whether GPU acces is TRUE
source /proj/uppmax2025-3-5/private/max/anaconda3/etc/profile.d/conda.sh
conda activate nlp-gpu
python -c 'import torch; print(f" GPU available: {torch.cuda.is_available()}\n",
                                f"Device name: {torch.cuda.get_device_name(0)}\n",
                                f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")'

#CREATE TEMP PATHS
python - << 'EOF'
import nbformat
nb = nbformat.read("RnD_Project.ipynb", as_version=4)

with open("load-nmt.py", "w") as f:
    for cell in nb.cells:
        if 'load NMT' in cell.get('metadata', {}).get('tags', []):
            f.write(cell.source + "\n")

with open("load-llm.py", "w") as f:
    for cell in nb.cells:
        if 'load LLM' in cell.get('metadata', {}).get('tags', []):
            f.write(cell.source + "\n")

with open("run-nmt.py", "w") as f:
    for cell in nb.cells:
        if 'run NMT' in cell.get('metadata', {}).get('tags', []):
            f.write(cell.source + "\n")

with open("run-llm.py", "w") as f:
    for cell in nb.cells:
        if 'run LLM' in cell.get('metadata', {}).get('tags', []):
            f.write(cell.source + "\n")

with open("load-rm.py", "w") as f:
    for cell in nb.cells:
        if 'load RM' in cell.get('metadata', {}).get('tags', []):
            f.write(cell.source + "\n")

with open("run-rm.py", "w") as f:
    for cell in nb.cells:
        if 'run RM' in cell.get('metadata', {}).get('tags', []):
            f.write(cell.source + "\n")

with open("get-bleu.py", "w") as f:
    for cell in nb.cells:
        if 'get BLEU' in cell.get('metadata', {}).get('tags', []):
            f.write(cell.source + "\n")
EOF

                        #|‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|#
#TEMP PATHS             #|   Pre-masking   |   Post-masking   |#
L_NMT='load-nmt.py'     #|   1/1 Done      |   ---            |#
L_LLM='load-llm.py'     #|   1/1 Done      |   ---            |#
R_NMT='run-nmt.py'      #|   1/1 Done      |   0/1 Done       |#
R_LLM='run-llm.py'      #|   9/9 Done      |   0/9 Done       |#
L_RM='load-rm.py'       #|   0/1 Done      |   ---            |#
R_RM='run-rm.py'        #|   0/2 Done      |   0/2 Done       |#
BLEU='get-bleu.py'      #|   2/2 Done      |   0/2 Done       |#
                        #|____________________________________|#

#EXECUTE SCRIPTS
#--------LOAD MODELS--------#
python $L_NMT
python $L_LLM

#--------TRANSLATE--------#
python $R_NMT
for i in $(seq 0 8); do                                 # loops over temps
    if [[ $i -eq 0 ]]; then
        prfx_1=$(printf "%.2f" 0.01)                    # sets first iteration to 0.01 instead of 0.00
    else
        prfx_1=$(printf "%.2f" $(echo "$i*0.2" | bc))   # prefix 1 ranges from 0.2 up to 1.6; actual temps
    fi
    prfx_2=$(printf "%02d" $((i*2)))                    # prefix 2 ranges from 00 to 16; used for names
    python $R_LLM $prfx_1 $prfx_2
done

#--------BLEU SCORE--------#
python $BLEU nmt ''
for i in $(seq 0 8); do                                # loops over temps
    prfx=$(printf "%02d" $((i*2)))                     # prefix same as prefix 2 in previous loop
    python $BLEU llm $prfx
done

#--------REWARD MODEL--------#
#python $L_RM
#python $R_RM