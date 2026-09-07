#!/usr/bin/env bash
# Evaluate the pretrained Polygloss model under each metalanguage translation condition.
# Results land in experiments/polygloss_metalang/<condition>/<glottocode>/metrics.json
set -euo pipefail

config="experiments/polygloss_metalang/eval.cfg"
glottocode=gawa1247

# TODO: rename these to match how each translation was actually produced.
# `no_translation` is the base condition and must stay as-is; the rest must match the
# `translation_<cond>` / `metalanguage_<cond>` column names in the CSVs.
conditions=(no_translation en_orig en_literal en_llm ur)

for cond in "${conditions[@]}"
do
    echo "=== Condition: $cond ==="
    python3 run.py "$config" --overrides glottocode="$glottocode" translation_condition="$cond"
done
