#!/usr/bin/env bash
set -e
./download_data.sh
python src/preprocess.py
python src/train.py --models persistence ar_ridge qepignn lstm gcn_lstm gat_lstm abl_no_amp abl_heat abl_no_walk qepignn_hybrid \
       --seeds 0 1 2 --epochs 40 --out results.json
python src/stats.py
python src/uncertainty.py
python src/figures.py
