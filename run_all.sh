#!/usr/bin/env bash
# Reproduit toutes les sorties numériques et la figure de l'article.
set -e
cd "$(dirname "$0")/scripts"
echo "== Gap_bench_sanity ==";        python3 Gap_bench_sanity.py
echo "== Gap_bench_race1 ==";         python3 Gap_bench_race1.py
echo "== Gap_bench_race2 ==";         python3 Gap_bench_race2.py
echo "== Analyse stationnaire ==";    python3 Art1_analyse_stationnaire.py
echo "== Analyse El Nino ==";         python3 Art1_analyse_elnino.py
mkdir -p ../figures ../results
mv -f fig_elnino.pdf ../figures/ 2>/dev/null || true
mv -f elnino_numbers.json ../results/ 2>/dev/null || true
echo "OK — figure dans figures/, nombres dans results/."
