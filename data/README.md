# Data Directory

## Structure

- `raw/` - Original UAH-DriveSet files (gitignored)
- `processed/` - Preprocessed CSV files (gitignored)
- `features/` - Window features (gitignored)
- `sample/` - Small sample data for testing (committed)

## Download Dataset

Download UAH-DriveSet from:
http://www.robesafe.uah.es/personal/eduardo.romera/uah-driveset/

Place in `data/raw/` directory.

## Processing

Run:
```bash
python scripts/preprocess_driveset.py
```
