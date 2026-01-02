#!/bin/bash

LOG_FILE="full_experiment_log_cny.txt"
: > "$LOG_FILE"  # clears file before starting new experiment run

SENTIMENT_SOURCES=(
    "naive_prompt_label"
    # "naive_plus_converted_prompt_label"
    # "finbert_label"
)

NEWS_HOLD_TIMES=(120)

for src in "${SENTIMENT_SOURCES[@]}"; do
    for hold in "${NEWS_HOLD_TIMES[@]}"; do

        echo "==========================================" | tee -a "$LOG_FILE"
        echo "Running experiment → SRC=$src | HOLD=$hold" | tee -a "$LOG_FILE"
        echo "==========================================" | tee -a "$LOG_FILE"

        python run_trading_strategy.py \
            --wallet_a 1000000 \
            --wallet_b 7130000 \
            --model_name chronos \
            --input_chunk_length 16 \
            --fx_data_path_train dataset/fx/usdcny-fx-train.csv \
            --fx_data_path_val dataset/fx/usdcny-fx-val.csv \
            --fx_data_path_test dataset/fx/usdcny-fx-test.csv \
            --news_data_path_train dataset/news/usdcny-news-train.csv \
            --news_data_path_test dataset/news/usdcny-news-test.csv \
            --use_frac_kelly \
            --n_epochs 50 \
            --output_dir results/graphs \
            --allow_news_overlap \
            --news_hold_minutes "$hold" \
            --sentiment_source "$src" \
        2>&1 | tee -a "$LOG_FILE"

        echo "" | tee -a "$LOG_FILE"
    done
done
