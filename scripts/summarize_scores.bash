#!/usr/bin/bash

# Bash strict mode
set -euo pipefail
IFS=$'\n\t'

printf "average weighted_acc\t%0.3f\n" $(grep weighted_accuracy results/*.txt | cut -f 3 -d ' ' | datamash mean 1)
printf "average weighted_F1\t%0.3f\n" $(grep weighted_F1 results/*.txt | cut -f 3 -d ' ' | datamash mean 1)
printf "average weighted_bal_acc\t%0.3f\n" $(grep weighted_bal_acc results/*.txt | cut -f 3 -d ' ' | datamash mean 1)
printf "average weighted_cal_F1\t%0.3f\n" $(grep weighted_cal_F1 results/*.txt | cut -f 3 -d ' ' | datamash mean 1)
printf "average noise F1\t%0.3f\n" $(grep '^0' results/*.txt | cut -f 3 | grep -v nan | datamash mean 1)
printf "average bathy F1\t%0.3f\n" $(grep '^40' results/*.txt | cut -f 3 | grep -v nan | datamash mean 1)
printf "average surface F1\t%0.3f\n" $(grep '^41' results/*.txt | cut -f 3 | grep -v nan | datamash mean 1)
