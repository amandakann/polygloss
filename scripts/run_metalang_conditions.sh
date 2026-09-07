#!/usr/bin/env bash
# Evaluate the pretrained Polygloss model under each metalanguage translation condition.
# Results land in experiments/polygloss_metalang/<condition>/<glottocode>/metrics.json
set -euo pipefail

config="experiments/polygloss_metalang/eval.cfg"
glottocode=gawa1247

conditions=(no_translation gls_en gls_ur lit_en gtr_en)

for cond in "${conditions[@]}"
do
    echo "=== Condition: $cond ==="
    python3 run.py "$config" --overrides glottocode="$glottocode" translation_condition="$cond"
done
