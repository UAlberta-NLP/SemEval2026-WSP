from deep_translator import MyMemoryTranslator
import time
import csv

def translate_with_mymemory(text, translator):
    """
    Translate text using MyMemory via DeepTranslator

    Args:
        text: Text to translate
        translator: MyMemoryTranslator instance

    Returns:
        Translated text or error message
    """
    try:
        if not text or text.strip() == '':
            return ''
        translation = translator.translate(text)
        return translation
    except Exception as e:
        print(f"\nTranslation error for '{text[:50]}...': {e}")
        return f"[TRANSLATION FAILED: {str(e)[:50]}]"

def main():
    input_file = 'dev-synsets-zh.tsv'
    output_file = 'dev-synsets-zh.tsv'

    # Initialize translator with email for higher limit (50k chars/day instead of 5k)
    your_email = "tling4@ualberta.ca"

    print(f"Initializing MyMemory translator (email: {your_email})...")
    translator = MyMemoryTranslator(
        source='en-US',  # English
        target='zh-CN',  # Simplified Chinese
        email=your_email
    )

    # Read the TSV file
    print(f"\nReading {input_file}...")
    rows = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        rows = list(reader)

    print(f"Loaded {len(rows)} rows")

    # Find column indices
    try:
        ex1_idx = header.index('example sentence 1')
        ex2_idx = header.index('example sentence 2')
        trans1_idx = header.index('example translation 1')
        trans2_idx = header.index('example translation 2')
    except ValueError as e:
        print(f"Error: Could not find required columns: {e}")
        print(f"Available columns: {header}")
        return

    # Calculate total sentences to translate
    total_sentences = 0
    total_chars = 0
    for row in rows:
        if len(row) > ex1_idx and row[ex1_idx].strip():
            total_sentences += 1
            total_chars += len(row[ex1_idx])
        if len(row) > ex2_idx and row[ex2_idx].strip():
            total_sentences += 1
            total_chars += len(row[ex2_idx])

    print(f"\nTotal sentences to translate: {total_sentences}")
    print(f"Total characters: {total_chars:,}")
    print(f"Daily limit (with email): 50,000 chars")

    if total_chars > 50000:
        print("\nWARNING: Total exceeds daily limit!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    # Translate sentences
    print(f"\nTranslating sentences using MyMemory API...")
    print("This may take a while due to API rate limits...")

    translated_count = 0
    chars_translated = 0

    for i, row in enumerate(rows, 1):
        # Ensure row has enough columns
        while len(row) <= max(trans1_idx, trans2_idx):
            row.append('')

        # Translate example sentence 1
        if len(row) > ex1_idx and row[ex1_idx].strip():
            chars_translated += len(row[ex1_idx])
            print(f"Translating {translated_count + 1}/{total_sentences} ({chars_translated:,}/{total_chars:,} chars)...", end='\r')

            translation = translate_with_mymemory(row[ex1_idx], translator)
            row[trans1_idx] = translation
            translated_count += 1

            # Rate limiting
            if translated_count % 10 == 0:
                time.sleep(1)
            else:
                time.sleep(0.3)

        # Translate example sentence 2
        if len(row) > ex2_idx and row[ex2_idx].strip():
            chars_translated += len(row[ex2_idx])
            print(f"Translating {translated_count + 1}/{total_sentences} ({chars_translated:,}/{total_chars:,} chars)...", end='\r')

            translation = translate_with_mymemory(row[ex2_idx], translator)
            row[trans2_idx] = translation
            translated_count += 1

            # Rate limiting
            if translated_count % 10 == 0:
                time.sleep(1)
            else:
                time.sleep(0.3)

    print(f"\nCompleted {translated_count} translations ({chars_translated:,} chars)")

    # Write the updated TSV file
    print(f"\nWriting updated file to {output_file}...")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        writer.writerows(rows)

    print("Translation complete!")
    print(f"\nUpdated {output_file} with translations in columns:")
    print(f"  - {header[trans1_idx]}")
    print(f"  - {header[trans2_idx]}")

    # Display first few examples
    print("\nFirst 3 translation examples:")
    for i in range(min(3, len(rows))):
        row = rows[i]
        if len(row) > ex1_idx and row[ex1_idx].strip():
            print(f"\nExample {i+1}:")
            print(f"  EN: {row[ex1_idx][:80]}")
            print(f"  ZH: {row[trans1_idx][:80]}")

if __name__ == "__main__":
    main()
