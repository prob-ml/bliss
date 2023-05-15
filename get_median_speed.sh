#!/home/zhteoh/bin/zsh

OUTPUT_FILE=$1
if [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: $0 <output_file>"
    exit 1
fi

echo "Getting median speed (it/s)..."
sed -e 's/.* \(.*\)it\/s.*/\1/g' -e 's/\(.*\)it\/s.*/\1/g' -e 's/Validation.*//g' -e 's/\?//g' -e 's/.*Epoch.*//g' -e '/^$/d' $OUTPUT_FILE | sort -n | uniq | awk '{a[NR]=$1} END{print a[int(NR/2 + 0.5)]}'
