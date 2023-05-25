#!/home/zhteoh/bin/zsh

OUTPUT_FILE=$1
if [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: $0 <output_file>"
    exit 1
fi

echo -n "Best validation loss: "
sed '/global step/!d' $OUTPUT_FILE | sed -e 's/[^\[]*\[A\(.*\)/\1/g' | sed '/not in top 1/d' | tail -1 | sed -e 's/^.*\((best \(.*\))\).*$/\2/'