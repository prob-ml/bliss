#!/home/zhteoh/bin/zsh

ALL_EXPS_LOGS_FILE=$1
if [ -z "$ALL_EXPS_LOGS_FILE" ]; then
    echo "Usage: $0 <log file of multirun experiment>"
    exit 1
fi

# Default: count occurrences of ".*#.*" in file
N_EXPS=${2:-$(grep -c "#.*" $ALL_EXPS_LOGS_FILE)}

for i in $(seq 0 $((N_EXPS-1))); do
    # First extract out logs for experiment i to a tmp file
    sed '/#'"$((i+1))"'/,$d' $ALL_EXPS_LOGS_FILE | sed '/#'"$i"'/,$!d' | sed -e 's/^.*\(#.*$\)/\1/g' > /tmp/exp$i.log
    # print first line of tmp file
    echo -n "EXPERIMENT: " && sed -n '1p' /tmp/exp$i.log | sed -e 's/^.*\((best \(.*\))\).*$/\2/'
    ./get_median_speed.sh /tmp/exp$i.log
    ./get_best_val_loss.sh /tmp/exp$i.log
done
