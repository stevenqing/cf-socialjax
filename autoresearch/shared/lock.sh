#!/bin/bash
# File-locked append to shared results.tsv
# Usage: bash autoresearch/shared/lock.sh <line_to_append>
#
# Uses flock to prevent concurrent write corruption between GPU workers.

LOCKFILE="/home/shuqing/SocialJax/autoresearch/shared/.results.lock"
RESULTS="/home/shuqing/SocialJax/autoresearch/shared/results.tsv"

LINE="$1"

(
    flock -w 10 200 || { echo "ERROR: Could not acquire lock"; exit 1; }
    echo "$LINE" >> "$RESULTS"
) 200>"$LOCKFILE"
