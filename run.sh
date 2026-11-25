#!/usr/bin/env bash
# Trains (or finetunes) on one of the datasets, on the given GPUs. Output goes to
# trlog.txt (or --log FILE); checkpoints to ckpt/<train|Finetune>/<name>/, where the
# name defaults to the dataset (--name NAME keeps runs apart).
#
#   ./run.sh --dataset {RNA3DB|CASP16|CASP15} --gpu 0,1,2,3 [--finetune --ckpt PATH]
#   ./run.sh --dataset CASP15 --gpu 0,1,2,3 --resume ckpt/train/CASP15/last.ckpt
#   ./run.sh --dataset CASP15 --gpu 0,2,3 --name CASP15_bp --log trlog_casp15_bp.txt
#   ./run.sh --dataset CASP15 --gpu 0,1,3 --val --epochs 300 --ckpt-every 10
#
# --val trains on <dataset>_train_split.csv and validates on <dataset>_val.csv
# (made by scripts/make_val_split.py) instead of training on all of
# <dataset>_train.csv; checkpoints are then ranked by the validation loss.
# --train-csv / --val-csv set the CSVs explicitly (--val-csv alone also enables
# validation). With --finetune, --val uses the same split files: the *_FT.csv
# features only add `ss_pred`, which the model does not use.
# --epochs overrides the number of epochs, --ckpt-every N keeps every N-th epoch's
# checkpoint (instead of only the 10 best).
#
#   ./run.sh --dataset RNA3DB --gpu 0,1,2,3 --name RNA3DB_bp_pre \
#       --train-csv metadata/RNA3DB_train_split_noCASP15.csv \
#       --val-csv metadata/RNA3DB_val.csv --epochs 1000 --ckpt-every 50
#   ./run.sh --dataset CASP15 --gpu 0,1,2,3 --finetune --ckpt PATH --val
set -euo pipefail
ulimit -n 65000

DATASET=""
GPU=""
CKPT=""
RESUME=""
RUN_NAME=""
LOG="trlog.txt"
FINETUNE=false
VAL=false
EPOCHS=""
CKPT_EVERY=""
TRAIN_CSV=""
VAL_CSV=""

TRAIN_EPOCHS=1500
FT_EPOCHS=200
FT_LR="0.00001"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)  DATASET="$2"; shift 2 ;;
    --gpu)      GPU="$2"; shift 2 ;;
    --ckpt)     CKPT="$2"; shift 2 ;;
    --finetune) FINETUNE=true; shift ;;
    --resume)   RESUME="$2"; shift 2 ;;
    --name)     RUN_NAME="$2"; shift 2 ;;
    --log)      LOG="$2"; shift 2 ;;
    --val)      VAL=true; shift ;;
    --epochs)   EPOCHS="$2"; shift 2 ;;
    --ckpt-every) CKPT_EVERY="$2"; shift 2 ;;
    --train-csv) TRAIN_CSV="$2"; shift 2 ;;
    --val-csv)  VAL_CSV="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

[[ -n "$DATASET" && -n "$GPU" ]] || {
  echo "Usage: $0 --dataset {RNA3DB|CASP16|CASP15} --gpu 0,1,2,3 [--finetune --ckpt PATH]"
  exit 1
}

# project name depends on mode
PROJ="train"
[[ "$FINETUNE" == "true" ]] && PROJ="Finetune"

# choose CSV based on dataset + mode
case "$DATASET" in
  RNA3DB)
    if [[ "$FINETUNE" == "true" ]]; then
      echo "RNA3DB finetune not supported in this script."
      exit 1
    fi
    CSV="metadata/RNA3DB_train.csv"
    NAME="RNA3DB"
    ;;
  CASP16)
    CSV="metadata/CASP16_train.csv"
    [[ "$FINETUNE" == "true" ]] && CSV="metadata/CASP16_FT.csv"
    NAME="CASP16"
    ;;
  CASP15)
    CSV="metadata/CASP15_train.csv"
    [[ "$FINETUNE" == "true" ]] && CSV="metadata/CASP15_FT.csv"
    NAME="CASP15"
    ;;
  *)
    echo "Usage: $0 --dataset {RNA3DB|CASP16|CASP15} --gpu 0,1,2,3 [--finetune --ckpt PATH]"
    exit 1
    ;;
esac

if [[ "$VAL" == "true" ]]; then
  CSV="metadata/${DATASET}_train_split.csv"
  [[ -n "$VAL_CSV" ]] || VAL_CSV="metadata/${DATASET}_val.csv"
  NAME="${NAME}_val"
fi
[[ -n "$TRAIN_CSV" ]] && CSV="$TRAIN_CSV"
for f in "$CSV" $VAL_CSV; do
  [[ -f "$f" ]] || { echo "$f missing (see scripts/make_val_split.py)"; exit 1; }
done

[[ -n "$RUN_NAME" ]] && NAME="$RUN_NAME"

NUM_DEVICES=$(awk -F',' '{print NF}' <<< "$GPU")

MAX_EPOCHS="$TRAIN_EPOCHS"
[[ "$FINETUNE" == "true" ]] && MAX_EPOCHS="$FT_EPOCHS"
[[ -n "$EPOCHS" ]] && MAX_EPOCHS="$EPOCHS"

CMD=(python3 train.py
  data_cfg.csv_path="$CSV"
  experiment.wandb.project="$PROJ"
  experiment.wandb.name="$NAME"
  experiment.trainer.devices="$NUM_DEVICES"
  experiment.trainer.max_epochs="$MAX_EPOCHS"
  # ~7x faster steps; set to True to locate the op behind a NaN
  experiment.trainer.detect_anomaly=False
)

if [[ -n "$VAL_CSV" ]]; then
  CMD+=(data_cfg.val_csv_path="$VAL_CSV" experiment.checkpointer.monitor=valid/loss)
fi
if [[ -n "$CKPT_EVERY" ]]; then
  CMD+=(experiment.checkpointer.save_top_k=-1 experiment.checkpointer.every_n_epochs="$CKPT_EVERY")
fi

if [[ "$FINETUNE" == "true" ]]; then
  [[ -n "$CKPT" ]] || { echo "--finetune requires --ckpt PATH"; exit 1; }
  CMD+=(experiment.finetune=true "experiment.warm_start=\"${CKPT}\"" experiment.optimizer.lr="$FT_LR")
elif [[ -n "$RESUME" ]]; then
  # Continues the run in place: weights, optimizer state, epoch and step count
  CMD+=(experiment.finetune=false "experiment.warm_start=\"${RESUME}\"")
else
  CMD+=(experiment.finetune=false experiment.warm_start=null)
fi

CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}" > "$LOG"
