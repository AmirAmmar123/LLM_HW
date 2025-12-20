#!/bin/bash

DOCS_DIR="knesset_protocols"
SENTENCES_FILE="/home/amir/LLM/HW2/knesset_sentences.txt"

while IFS= read -r sentence; do
    # דילוג על שורה ריקה
    [[ -z "$sentence" ]] && continue

    # הסרת נקודה בסוף המשפט (אם קיימת)
    sentence="${sentence%.}"

    # יצירת regex גמיש לרווחים
    regex=$(echo "$sentence" | sed 's/[[:space:]]\+/[[:space:]]\\+/g')

    found=0

    for doc in "$DOCS_DIR"/*.docx; do
        if unzip -p "$doc" | grep -qiE "$regex"; then
            echo "✔ FOUND:"
            echo "   Sentence: $sentence"
            echo "   File: $(basename "$doc")"
            found=1
            break
        fi
    done

    if [[ $found -eq 0 ]]; then
        echo "✘ NOT FOUND:"
        echo "   Sentence: $sentence"
    fi

    echo "--------------------------------------------"
done < "$SENTENCES_FILE"
