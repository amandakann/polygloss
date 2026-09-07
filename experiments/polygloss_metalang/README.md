# Metalanguage translation conditions

Measures how much the metalanguage translation line in a Polygloss prompt contributes to
glossing quality, and whether the kind of translation matters. Every prompt template
contains:

```
Translation in $metalang: $translation
```

Each condition varies only that line; the transcription, language name, task format and model
are held fixed, and all conditions run over an identical row set from one CSV source.

## Conditions

| `translation_condition` | Renders as | Source columns |
|---|---|---|
| `no_translation` | `Translation in English: None` | — (base condition) |
| `en_orig` | `Translation in English: …` | `translation_en_orig`, `metalanguage_en_orig` |
| `en_literal` | `Translation in English: …` | `translation_en_literal`, `metalanguage_en_literal` |
| `en_llm` | `Translation in English: …` | `translation_en_llm`, `metalanguage_en_llm` |
| `ur` | `Translation in Urdu: …` | `translation_ur`, `metalanguage_ur` |


The base condition reproduces the pretraining fallback (`Translation in English: None`) rather
than deleting the line, so a score drop reflects the missing translation rather than an
off-distribution prompt format.

Note the distinction between `translation_condition` unset (uses the plain `translation`
column — pre-experiment behaviour, and what every other experiment folder does) and
`translation_condition=no_translation` (deliberately withholds the translation).

## Data requirements

`local_dataset_path` must point at a directory with `train/`, `dev/`, `test/` subdirectories of
CSVs carrying, in addition to the usual columns, one pair per condition:

```
translation_<cond>,metalanguage_<cond>
```

- `metalanguage_<cond>` must be written into the CSV directly (`English` / `Urdu`). Nothing
  derives it at load time, so a blank cell renders `Translation in an unknown language:`.
- Every CSV within a split must carry the identical column set — HuggingFace `datasets`
  concatenates them, and one file missing a column fails the whole load.
- `metalang_glottocode_<cond>` (`stan1293` / `urdu1245`) is optional provenance; the
  training/eval pipeline never reads it.


## Running

```bash
bash scripts/run_metalang_conditions.sh
```

or a single condition:

```bash
python run.py experiments/polygloss_metalang/eval.cfg \
    --overrides glottocode=gawa1247 translation_condition=ur
```

Results: `experiments/polygloss_metalang/<condition>/<glottocode>/metrics.json`, plus one WandB
run per condition (`translation_condition` is logged as a config field for grouping).