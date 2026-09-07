#!/usr/bin/env bash
# Train + evaluate low-rank adapters for Gawarbati across
# metalang translation conditions and train data sizes.
set -euo pipefail

config="experiments/polygloss_gwb/lora.cfg"
data_root="/home/kann/data"
glottocode=gawa1247

conditions=(no_translation gls_en gls_ur lit_en gtr_en)
sizes=(s m l)

for cond in "${conditions[@]}"
do
    for size in "${sizes[@]}"
    do
        echo "=== Condition: $cond | Adapter size: $size ==="
        python3 run.py "$config" \
            --overrides \
                glottocode="$glottocode" \
                translation_condition="$cond" \
                local_dataset_path="$data_root/gawarbati-$size"
    done
done
