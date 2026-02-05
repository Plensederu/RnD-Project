#!/bin/bash
#SBATCH -A uppmax2025-3-5
#SBATCH -M pelle
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --mail-type=END
#SBATCH -t 20:00:00

# load module & package
module load GCC/12.3.0

export CUDA_VISIBLE_DEVICES
export PYTORCH_ALLOC_CONF=expandable_segments:True
export ACCESS_TOKEN=#replace with Hugging Face access token

# setup conda environment and test whether GPU acces is TRUE
source /path/to/your/conda/env
conda activate venv
hf auth login --token $ACCESS_TOKEN --add-to-git-credential

#CREATE TEMP PATHS
python - << 'EOF'
import nbformat
nb = nbformat.read('RnD_Project.ipynb', as_version=4)

with open('load-nmt.py', 'w') as f:
    for cell in nb.cells:
        if 'NMT' in cell.get('metadata', {}).get('tags', []):
            f.write(cell.source + '\n')

with open('load-llm.py', 'w') as f:
    for cell in nb.cells:
        if 'LLM' in cell.get('metadata', {}).get('tags', []):
            f.write(cell.source + '\n')

with open('load-rm.py', 'w') as f:
    for cell in nb.cells:
        if 'RM' in cell.get('metadata', {}).get('tags', []):
            f.write(cell.source + '\n')

with open('get-bleu.py', 'w') as f:
    for cell in nb.cells:
        if 'BLEU' in cell.get('metadata', {}).get('tags', []):
            f.write(cell.source + '\n')

with open('make-plot.py', 'w') as f:
    for cell in nb.cells:
        if 'PLOT' in cell.get('metadata', {}).get('tags', []):
            f.write(cell.source + '\n')

with open('make-csv.py', 'w') as f:
    for cell in nb.cells:
        if 'CSV' in cell.get('metadata', {}).get('tags', []):
            f.write(cell.source + '\n')

with open('clean-noise.py', 'w') as f:
    for cell in nb.cells:
        if 'NOISE' in cell.get('metadata', {}).get('tags', []):
            f.write(cell.source + '\n')

with open('align-sents.py', 'w') as f:
    for cell in nb.cells:
        if 'SELECT' in cell.get('metadata', {}).get('tags', []):
            f.write(cell.source + '\n')

with open('mask-words.py', 'w') as f:
    for cell in nb.cells:
        if 'MASK' in cell.get('metadata', {}).get('tags', []):
            f.write(cell.source + '\n')

with open('get-total.py', 'w') as f:
    for cell in nb.cells:
        if 'MASK' in cell.get('metadata', {}).get('tags', []):
            f.write(cell.source + '\n')

with open('stat-sign-test.py', 'w') as f:
    for cell in nb.cells:
        if 'SIGNIFICANCE' in cell.get('metadata', {}).get('tags', []):
            f.write(cell.source + '\n')
EOF

#                           If you wish to try out the code yourself, 
#                           I made this to keep track of which steps 
#                           are done for debugging and testing reasons: 

                            #|‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|#
#TEMP PATHS                 #|   Pre-masking   |   Post-masking   |#
NMT='load-nmt.py'           #|   0/1  Done     |   0/1  Done      |#
LLM='load-llm.py'           #|   0/9  Done     |   0/9  Done      |#
RM='load-rm.py'             #|   0/9  Done     |   0/9  Done      |#
TOTAL='get-total.py'        #|   0/1  Done     |   0/1  Done      |#
BLEU='get-bleu.py'          #|   0/10 Done     |   0/10 Done      |#
CSV='make-csv.py'           #|   0/1  Done     |   0/1  Done      |#
PLOT='make-plot.py'         #|   ---           |   0/1  Done      |#
NOISE='clean-noise.py'      #|   0/9  Done     |   0/9  Done      |#
ALIGN='align-sents.py'      #|   0/2  Done     |   0/2  Done      |#
MASK='mask-words.py'        #|   0/1  Done     |   ---            |#
STT='stat-sign-test.py'     #|   0/1  Done     |   0/1  Done      |#
                            #|____________________________________|#

#----------------------------BEFORE MASKING----------------------------#
#--------LOAD MODELS--------#
python $NMT ''
for i in $(seq 0 8); do
    if [[ $i -eq 0 ]]; then
        prfx_1=$(printf '%.2f' 0.01)
    else
        prfx_1=$(printf '%.2f' $(echo '$i*0.2' | bc))
    fi
    prfx_2=$(printf '%02d' $((i*2)))
    python $LLM $prfx_1 $prfx_2 ''
done

#--------REMOVE NOISE--------#
for i in $(seq 0 8); do
    temp=$(printf 't%02d' $((i*2)))
    python $NOISE $temp ''
done

#--------ALIGN SENTENCES--------#
for i in $(seq 0 8); do
    temp=$(printf 't%02d' $((i*2)))
    python $ALIGN llm $temp ''
done
for i in $(seq 4); do
    python $ALIGN ref $i
done

#--------BLEU SCORE--------#
python $BLEU nmt '' ''
for i in $(seq 0 8); do
    prfx=$(printf 't%02d_' $((i*2)))
    python $BLEU llm $prfx ''
done
python $CSV ''

#--------REWARD MODEL--------#
for i in $(seq 0 8); do
    temp=$(printf 't%02d' $((i*2)))
    python $RM $temp ''
done
python $TOTAL 
python $SST ''

#--------PERTURBATION--------#
python $MASK

#----------------------------AFTER MASKING----------------------------#
#--------LOAD MODELS--------#
python $NMT '-mask'
for i in $(seq 0 8); do
    if [[ $i -eq 0 ]]; then
        prfx_1=$(printf '%.2f' 0.01)
    else
        prfx_1=$(printf '%.2f' $(echo '$i*0.2' | bc))
    fi
    prfx_2=$(printf '%02d' $((i*2)))
    python $LLM $prfx_1 $prfx_2 '-mask'
done

#--------REMOVE NOISE--------#
for i in $(seq 0 8); do
    temp=$(printf 't%02d' $((i*2)))
    python $NOISE $temp '-mask'
done

#--------ALIGN SENTENCES--------#
for i in $(seq 0 8); do
    temp=$(printf 't%02d' $((i*2)))
    python $ALIGN llm $temp ''
done

#--------BLEU SCORE--------#
python $BLEU nmt '' '-mask'
for i in $(seq 0 8); do
    prfx=$(printf 't%02d_' $((i*2)))
    python $BLEU llm $prfx '-mask'
done
python $CSV '-mask'

#--------REWARD MODEL--------#
for i in $(seq 0 8); do
    temp=$(printf 't%02d' $((i*2)))
    python $RM $temp 'mask-'
done
python $TOTAL 'mask-'
python $SST '-mask'

#--------PLOT BLEU--------#
python $PLOT
