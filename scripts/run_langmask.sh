config="experiments/polygloss_gwb/eval.cfg"
language_mask=Nganasan

echo "Masking language as $language_mask"

python3 run.py $config --overrides glottocode=gawa1247 language_mask=$language_mask
for lora in l m s
do
    echo "Running with LoRA-$lora"
    python run.py $config --overrides glottocode=gawa1247 adapter_dir="experiments/polygloss_gwb/gawa1247/lora_gwb_$lora.model" language_mask=$language_mask
done