# Resume Matcher Application - Build Guide

## The Problem

The error you encountered:
```
OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a Python package or a valid path to a data directory.
```

This happened because PyInstaller has trouble bundling spaCy models correctly. The spaCy model `en_core_web_sm` contains many files and dependencies that need to be properly included in the executable.

## The Solution

I've **replaced spaCy with RAKE-NLTK** to eliminate this problem entirely. RAKE-NLTK is:
- Pure Python (no complex model bundling needed)
- Lightweight and fast
- Works "out of the box" with PyInstaller
- Provides excellent keyword extraction capabilities

### 1. Updated `app.py`
- Replaced spaCy with RAKE-NLTK for keyword extraction
- Added automatic NLTK stopwords download
- Simplified dependencies and removed complex model loading

### 2. Updated `app.spec`
- Removed spaCy model data bundling
- Added RAKE-NLTK and NLTK to hidden imports
- Much simpler and more reliable configuration

### 3. `setup_rake_nltk.py`
- Standalone script to ensure RAKE-NLTK is installed
- Downloads NLTK stopwords automatically
- Verifies the installation works correctly

### 4. `build_app.py`
- Complete build automation script
- Handles RAKE-NLTK setup, cleaning, and PyInstaller building
- Provides detailed feedback during the build process

## How to Fix

### Option 1: Use the Build Script (Recommended)
```bash
cd basicfile
python build_app.py
```

This script will:
1. Install and verify RAKE-NLTK dependencies
2. Download NLTK stopwords
3. Clean previous build artifacts
4. Build the PyInstaller executable
5. Verify the build

### Option 2: Manual Steps
```bash
# 1. Install RAKE-NLTK
pip install rake-nltk

# 2. Download NLTK stopwords
python -c "import nltk; nltk.download('stopwords')"

# 3. Verify installation
python setup_rake_nltk.py

# 4. Build with PyInstaller
pyinstaller app.spec
```

### Option 3: Simple PyInstaller Build
Since RAKE-NLTK is pure Python, you can also build directly:
```bash
pyinstaller --onefile --hidden-import rake_nltk --hidden-import nltk app.py
```

## Troubleshooting

### If the build still fails:
1. **Check RAKE-NLTK installation**: `python -c "from rake_nltk import Rake; print('RAKE-NLTK works')"`
2. **Verify PyInstaller**: `pyinstaller --version`
3. **Clean and rebuild**: Delete `build/`, `dist/`, and `__pycache__/` directories
4. **Use virtual environment**: Ensure all dependencies are in the same environment

### Common Issues:
- **RAKE-NLTK not found**: Run `pip install rake-nltk`
- **NLTK stopwords missing**: Run `python -c "import nltk; nltk.download('stopwords')"`
- **Import errors**: Check that RAKE-NLTK and NLTK are in `hiddenimports`
- **Small executable**: RAKE-NLTK is lightweight (~1-2MB total)

## File Structure After Build
```
basicfile/
├── app.py              # Main application (updated)
├── app.spec            # PyInstaller spec (updated)
├── setup_spacy.py      # spaCy setup script
├── build_app.py        # Build automation script
├── dist/
│   └── app            # Your executable
└── build/             # Build artifacts
```

## Testing the Build
After successful build, test the executable:
```bash
./dist/app
```

The application should start without the spaCy model error and display the Flask web interface.

## Notes
- RAKE-NLTK is lightweight (~1-2MB), so the final executable will be much smaller
- The build process is faster since no large models need to be bundled
- RAKE-NLTK provides excellent keyword extraction for job descriptions
- No complex model dependencies to manage 