#!/bin/bash

rsync -avxH train_test.py soulej@bigfoot.ciment:/bettik/soulej/omarl_experiments/

if [ -e ./tmp_job.sh ]; then
    rm -rf tmp_job.sh
fi

WALLTIME=$1
if [ -z "$1" ]; then
    WALLTIME="0:30:00"
fi

cat <<EOT > tmp_job.sh
#!/bin/bash

#OAR -n Tests
#OAR -l /nodes=1/gpu=1,walltime=$WALLTIME
#OAR --stdout %jobid%.out
#OAR --stderr %jobid%.err
#OAR --project pr-ai4cmas
#OAR -p gpumodel='V100'

echo "Job started at $(date)"
echo "Current directory: $(pwd)"
echo "List of files in current directory:"
ls -la

if [ -e "/applis/environments/conda.sh" ]; then
    source /applis/environments/conda.sh
else
    source ~/anaconda3/etc/profile.d/conda.sh
fi

conda info --envs
echo "Activating marllib environment"
conda activate marllib

export PYTHONPATH="${PYTHONPATH}:./../../../mma_wrapper"

if [ $? -ne 0 ]; then
    echo "Error during conda activation"
    exit 1
fi

conda info --envs

which python
python --version

echo "Starting train_test.py"
python -u train_test.py

if [ $? -ne 0 ]; then
    echo "Error during train_test.py execution"
    exit 1
fi

echo "Job ended at $(date)"
EOT

chmod +x tmp_job.sh
rsync -avxH tmp_job.sh soulej@bigfoot.ciment:/bettik/soulej/omarl_experiments
ssh soulej@bigfoot.ciment -t "cd /bettik/soulej/omarl_experiments/ ; rm -rf *.err *.out exp_results ; oarsub -S ./tmp_job.sh"
mkdir exp_results

convert_to_seconds() {
    local TIME=$1
    local IFS=:
    local PARTS=($TIME)
    local HOURS=${PARTS[0]}
    local MINUTES=${PARTS[1]}
    local SECONDS=${PARTS[2]}
    echo $((HOURS * 3600 + MINUTES * 60 + SECONDS))
}

show_progress_bar() {
  local DURATION=$1
  local INTERVAL=1
  local ELAPSED=0
  local BAR_SIZE=50

  while [ $ELAPSED -lt $DURATION ]; do
    local PROGRESS=$(( ELAPSED * BAR_SIZE / DURATION ))
    local PERCENT=$(( ELAPSED * 100 / DURATION ))
    local BAR=''

    for ((i=0; i<PROGRESS; i++)); do
      BAR+='='
    done

    for ((i=PROGRESS; i<BAR_SIZE; i++)); do
      BAR+=' '
    done

    printf "\r[%s] %d%%" "$BAR" "$PERCENT"

    sleep $INTERVAL
    ELAPSED=$(( ELAPSED + INTERVAL ))
  done

  printf "\r[%s] 100%%\n" $(printf '=%.0s' {1..50})
}

########################

check_substring() {
    local file=$1
    local substring="W soulej"
    
    if grep -q "$substring" "$file"; then
        # found waiting in stats
        return 0
    else
        return 1
    fi
}

echo "WAITING FOR TRAINING TO START..."

while true; do
  ssh soulej@bigfoot.ciment -t "cd /bettik/soulej/omarl_experiments ; oarstat -u soulej > stats.log ;"
  rsync -avxH soulej@bigfoot.ciment:/bettik/soulej/omarl_experiments/stats.log .
  if ! check_substring "stats.log"; then
    ssh soulej@bigfoot.ciment -t "cd /bettik/soulej/omarl_experiments ; rm -rf stats.log ;"
    rm -rf stats.log
    sleep 5
    echo "TRAINING STARTED!"
    break
  fi
  sleep 60
done

########################

echo "WAITING FOR TRAINING TO FINISH..."

SECONDS_DURATION=$(convert_to_seconds $WALLTIME)
# sleep $SECONDS_DURATION + 60
show_progress_bar $SECONDS_DURATION + 60

echo -e "\nTRAINING FINISHED!"

rsync -avxH soulej@bigfoot.ciment:/bettik/soulej/omarl_experiments/exp_results/ ./exp_results/
ssh soulej@bigfoot.ciment -t "cd /bettik/soulej/omarl_experiments ; rm -rf tmp_job.sh"
rm -rf tmp_job.sh
