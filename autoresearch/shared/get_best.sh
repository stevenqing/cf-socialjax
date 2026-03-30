#!/bin/bash
# Get the best known result for an environment from shared results.
# Usage: bash autoresearch/shared/get_best.sh <env_name>
# Returns: the best returned_episode_returns value

ENV_NAME="$1"
RESULTS="/home/shuqing/SocialJax/autoresearch/shared/results.tsv"

grep "$ENV_NAME" "$RESULTS" | \
    grep -v "^#" | \
    awk -F'\t' '{print $3}' | \
    grep -v "TIMEOUT\|CRASHED" | \
    sort -rn | \
    head -1
