from pathlib import Path

import pandas as pd

import wandb
from data.model import load_igt
from src.evaluation.evaluate import evaluate

modelname = "GLM0_Nganasan"

wandb.init(project="typgloss", entity="amanda-kann-stockholm-university", name=modelname)

all_preds = []
evaluation_languages = [
    "MeraKhan",
    "Jawarya"]
evaluation_isocodes = [
    "MeraKhan",
    "Jawarya"]



for glotto, iso in zip(evaluation_languages, evaluation_isocodes):
    preds = load_igt(
        Path(__file__).parent / "pred" / f"{iso}_{modelname}",
        id_prefix=iso,
        source="liljegren",
    )
    gold = load_igt(
        Path(__file__).parent / "gold" / f"{iso}_gold",
        id_prefix=iso,
        source="liljegren",
    )
    for p, g in zip(preds, gold):
        all_preds.append(
            {
                "predicted": p.glosses,
                "reference": g.glosses,
                "task": "t2g",
                "id": p.id,
                "glottocode": glotto,
            }
        )
        # all_preds.append(
        #     {
        #         "predicted": p.segmentation,
        #         "reference": g.segmentation,
        #         "task": "t2s",
        #         "id": p.id,
        #         "glottocode": glotto,
        #     }
        # )

df = pd.DataFrame(all_preds)
wandb.log({"predictions": wandb.Table(dataframe=df)})
metrics = evaluate(df)
wandb.log(data={"test": metrics})
