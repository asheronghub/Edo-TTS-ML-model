import pandas as pd
import os
import re

# IMPORTANT: This list of characters MUST exactly match the 'characters' string
# you defined in your edo_tts_config.json file, excluding <PAD>, <EOS>, <BOS>.
# It includes the hyphen and space as per your definition.
_EDO_VALID_CHARS = "-abdefghiklmnoprstuvwyz àáèéìíòóùú̀ẹọgbghkhkpmwrhrrvb "

# Create a regex pattern to find characters NOT in our valid set.
# This will be used for a WARNING only, not removal, to help you verify your char set.
_INVALID_CHARS_REGEX = re.compile(f'[^{re.escape(_EDO_VALID_CHARS)}]')


def custom_edo_cleaner(text):
    """
    A cleaner for Edo text based on your constraints:
    - Normalizes whitespace.
    - Converts text to lowercase.
    - Includes a warning if unexpected characters are found based on _EDO_VALID_CHARS.
    """
    text = text.strip()  # Remove leading/trailing whitespace
    text = text.lower()  # Convert to lowercase for consistency

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # --- IMPORTANT VERIFICATION STEP ---
    # This part is for *checking* if any unexpected characters exist.
    # It does NOT remove characters, but warns you if your _EDO_VALID_CHARS list is incomplete.
    found_invalid_chars = _INVALID_CHARS_REGEX.findall(text)
    if found_invalid_chars:
        print(f"WARNING: Found unexpected characters in '{text}': {set(found_invalid_chars)}. "
              f"Ensure your edo_tts_config.json 'characters' list is complete.", file=os.sys.stderr)
    # --- END VERIFICATION ---

    return text

# --- Script to clean your metadata.csv ---
def clean_metadata_csv(metadata_path='metadata.csv'):
    """
    Loads metadata.csv, applies custom_edo_cleaner to transcripts, and saves it back.
    """
    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found. Please ensure it's in the correct directory.", file=os.sys.stderr)
        return

    # Read the CSV without a header, assuming pipe separator
    df = pd.read_csv(metadata_path, sep='|', header=None, names=['audio_path', 'transcript'])

    print(f"Processing {len(df)} entries in '{metadata_path}'...")
    print("Original transcripts (first 3):")
    for i in range(min(3, len(df))):
        print(df['transcript'].iloc[i])

    # Apply the cleaner function to the transcript column
    df['transcript'] = df['transcript'].apply(custom_edo_cleaner)

    print("\nCleaned transcripts (first 3):")
    for i in range(min(3, len(df))):
        print(df['transcript'].iloc[i])

    # Save the cleaned data back to metadata.csv
    df.to_csv(metadata_path, sep='|', header=False, index=False)

    print(f"\n'{metadata_path}' has been cleaned and updated successfully.")
    print("Please check for any WARNINGs above about unexpected characters.")


# --- Uncomment the line below to run the cleaning when this script is executed ---
if __name__ == "__main__":
    clean_metadata_csv()