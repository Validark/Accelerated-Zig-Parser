#/bin/bash
if [ "$1" == "crlf" ] || [ "$1" == "lf" ]; then
    find ./src/files_to_parse -type f -name '*.zig' | while read file; do
        if [ "$1" == "crlf" ]; then
            perl -pi -e 's/\r?\n/\r\n/g' "$file"
        else
            perl -pi -e 's/\r\n/\n/g' "$file"
        fi
    done
else
    cat << EOF
Usage: crlf.sh [crlf|lf]
EOF
   exit 1;
fi
