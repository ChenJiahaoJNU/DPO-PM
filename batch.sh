#!/bin/bash
set -e  

TRAINER_FILES=(
    # "trainers_oringinal.py"
    "dpo_Miin.py"
    # "trainers_spo.py"
    # "trainers_Call_DPO.py"
)


DATASET_PATHS=(
    # "/root/autodl-fs/2025autumn/HF_Datasets/HH1"
    # "/root/autodl-fs/2025autumn/HF_Datasets/HH2"
    # "/root/autodl-fs/2025autumn/HF_Datasets/HH3"
    # "/root/autodl-fs/2025autumn/HF_Datasets/HH4"
    # "/root/autodl-fs/2025autumn/HF_Datasets/HH5"
    "/root/autodl-fs/2025autumn/HF_Datasets/HH6"
    # "/root/autodl-fs/2025autumn/HF_Datasets/HH7"
    # "/root/autodl-fs/2025autumn/HF_Datasets/HH8"
    # "/root/autodl-fs/2025autumn/HF_Datasets/HH9"

)

LOG_DIR="train_logs"
mkdir -p "$LOG_DIR"

modify_dataset_path() {
    local dataset_path="$1"
    local datasets_file="/root/autodl-fs/2025autumn/Leaning/direct-preference-optimization-main/preference_datasets.py"
    
    cp "$datasets_file" "${datasets_file}.backup"
    
    sed -i "s|dataset_root = Path(\".*\")|dataset_root = Path(\"$dataset_path\")|g" "$datasets_file"

}
restore_dataset_file() {
    local datasets_file="/root/autodl-fs/2025autumn/Leaning/direct-preference-optimization-main/preference_datasets.py"
    
    if [ -f "${datasets_file}.backup" ]; then
        cp "${datasets_file}.backup" "$datasets_file"
        rm "${datasets_file}.backup"
    fi
}

trap restore_dataset_file EXIT

for dataset_path in "${DATASET_PATHS[@]}"; do
    dataset_name=$(basename "$dataset_path")
    
    echo -e "\n========================================"
    echo " $dataset_name ($dataset_path)"
    echo "Time: $(date '+%H:%M:%S')"
    echo "========================================"
    
    modify_dataset_path "$dataset_path"
    
    for trainer_file in "${TRAINER_FILES[@]}"; do
        trainer_name=$(basename "$trainer_file" .py)
        echo -e "\n----------------------------------------"
        echo "trainer: $trainer_file"
        echo "----------------------------------------"

        > trainers.py

        if [ ! -f "$trainer_file" ]; then
            continue
        fi
        cp "$trainer_file" trainers.py

        timestamp=$(date +%Y%m%d_%H%M%S)
        LOG_FILE="$LOG_DIR/${timestamp}_${dataset_name}_${trainer_name}.log"
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6  
        export PYTHONUNBUFFERED=1 

        echo "🚀 : $dataset_name, Log$LOG_FILE"
            python -u train.py \
            model=pythia28 \
            datasets=[hh] \
            loss=dpo \
            loss.beta=0.1 \
            exp_name="${dataset_name}_${trainer_name}_anthropic_dpo_pythia28" \
            gradient_accumulation_steps=2 \
            batch_size=56\
            eval_batch_size=32 \
            trainer=FSDPTrainer \
            sample_during_eval=false \
            model.fsdp_policy_mp=bfloat16 \
            2>&1 | tee "$LOG_FILE"

        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "✅ $trainer_file  $dataset_name "
        else
            echo "❌ $trainer_file  $dataset_name $LOG_FILE"
        fi
    done
done

echo -e "\n========================================"
echo "$LOG_DIR"
echo "========================================"