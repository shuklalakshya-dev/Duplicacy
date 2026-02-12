# Audio Duplicate Detection System

A powerful Python tool that identifies duplicate audio files using acoustic fingerprinting technology. Unlike simple file hash comparisons, this system analyzes the actual sound content, enabling detection of duplicates even across different formats, bitrates, and encodings.

## Features

- **Acoustic Fingerprinting**: Uses Chromaprint technology to analyze audio content
- **Format Agnostic**: Detects duplicates across MP3, FLAC, WAV, AAC, OGG, M4A, WMA, OPUS
- **Smart Recommendations**: Suggests which file to keep based on quality
- **Safe Operation**: Never auto-deletes; generates reviewable scripts
- **Detailed Reports**: JSON export for record-keeping
- **Space Savings**: Calculates potential disk space recovery

## Prerequisites

### 1. Python 3.7+

Ensure Python 3.7 or higher is installed:

```bash
python --version
```

### 2. Chromaprint (REQUIRED)

The system requires the `fpcalc` command-line tool from Chromaprint.

#### Windows Installation

1. Download from: https://acoustid.org/chromaprint
2. Extract the archive
3. Either:
   - Add the extracted folder to your system PATH, OR
   - Place `fpcalc.exe` in the same folder as `audio_duplicate_detector.py`, OR
   - Specify the path using `--fpcalc` argument

#### macOS Installation

```bash
brew install chromaprint
```

#### Linux Installation (Ubuntu/Debian)

```bash
sudo apt install libchromaprint-tools
```

#### Linux Installation (Fedora)

```bash
sudo dnf install chromaprint-tools
```

## Installation

1. Clone or download this repository:

```bash
cd C:\Users\MSI-PC\Duplicacy
```

2. (Optional) Create a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

3. No Python packages required - uses standard library only!

## Usage

### Interactive Mode

Simply run the script without arguments:

```bash
python audio_duplicate_detector.py
```

You'll be prompted to enter your music folder path.

### Command Line Mode

```bash
# Basic scan
python audio_duplicate_detector.py "C:\Music"

# Adjust similarity threshold (0.0-1.0)
python audio_duplicate_detector.py "C:\Music" --threshold 0.90

# Scan without subfolders
python audio_duplicate_detector.py "C:\Music" --no-recursive

# Specify output directory for reports
python audio_duplicate_detector.py "C:\Music" --output "C:\Reports"

# Specify fpcalc location
python audio_duplicate_detector.py "C:\Music" --fpcalc "C:\Tools\fpcalc.exe"

# Skip deletion script generation
python audio_duplicate_detector.py "C:\Music" --no-script
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `folder` | Path to music folder | (interactive prompt) |
| `-t, --threshold` | Similarity threshold (0.0-1.0) | 0.95 |
| `-r, --recursive` | Scan subfolders | True |
| `--no-recursive` | Don't scan subfolders | False |
| `-o, --output` | Output directory for reports | Current directory |
| `--fpcalc` | Path to fpcalc executable | Auto-detect |
| `--no-script` | Skip deletion script generation | False |
| `-v, --version` | Show version | - |

## Understanding the Output

### Console Output

The tool displays:
1. Scanning progress with file counts
2. Duplicate groups with file details
3. **[KEEP]** / **[DELETE]** recommendations
4. File sizes, formats, and durations
5. Total potential space savings

### Example Output

```
==============================================================
DUPLICATE DETECTION RESULTS
==============================================================

Total files scanned: 1234
Duplicate groups found: 15
Total duplicate files: 42
Potential space savings: 1.25 GB

--------------------------------------------------
DUPLICATE GROUP 1 (3 files)
Potential savings: 45.23 MB
--------------------------------------------------
  [KEEP]      C:\Music\Album\song.flac
              Size: 35.50 MB, Format: .FLAC
              Duration: 245.3s
  [DELETE]    C:\Music\Downloads\song.mp3
              Size: 12.30 MB, Format: .MP3
              Duration: 245.1s
  [DELETE]    C:\Music\Old\song_copy.mp3
              Size: 8.15 MB, Format: .MP3
              Duration: 245.2s
```

### Generated Files

After scanning, the tool creates:

1. **`duplicate_report_YYYYMMDD_HHMMSS.json`** - Detailed JSON report
2. **`delete_duplicates_YYYYMMDD_HHMMSS.bat`** - Windows deletion script
3. **`delete_duplicates_YYYYMMDD_HHMMSS.sh`** - Unix/macOS deletion script

## Safety Features

- **No Auto-Delete**: Files are NEVER automatically deleted
- **Commented Scripts**: Deletion commands are commented out by default
- **Review Required**: User must edit and uncomment lines to execute
- **Keep Recommendations**: Highest quality file is always recommended to keep
- **JSON Backup**: Full results saved for record-keeping

## How It Works

### Acoustic Fingerprinting

1. **Audio Extraction**: fpcalc decodes audio files and extracts raw audio
2. **Feature Analysis**: Chromaprint analyzes spectral features
3. **Fingerprint Generation**: Creates a compact acoustic fingerprint
4. **Similarity Comparison**: Bit-level comparison of fingerprints

### Quality Recommendations

Files are ranked by:
1. **Format** (Lossless > Lossy): FLAC > WAV > M4A > AAC > OPUS > OGG > MP3 > WMA
2. **File Size**: Larger usually means higher bitrate/quality
3. **Lossless Bonus**: Extra weight for FLAC and WAV files

### Similarity Threshold Guide

| Threshold | Use Case |
|-----------|----------|
| 1.00 | Exact audio matches only |
| 0.95 | Same song, different encodings (recommended) |
| 0.90 | Allow minor differences/edits |
| 0.85 | Catch different masters/versions |
| 0.80 | Liberal matching (may have false positives) |

## Troubleshooting

### "fpcalc not found"

- Ensure Chromaprint is installed
- Add fpcalc to PATH or use `--fpcalc` argument
- Place fpcalc.exe in the script directory

### "Fingerprinting failed" for some files

- File may be corrupted or too short
- Unsupported codec inside container
- System falls back to file hash comparison

### Slow Performance

- Large libraries take time (1000+ files)
- Consider scanning specific subfolders
- SSD storage significantly improves speed

### False Positives

- Increase threshold (e.g., `--threshold 0.98`)
- Very similar songs may match (remixes, covers)

## Use Cases

- **Clean Music Library**: Remove duplicate downloads
- **Format Migration**: Find MP3s that have FLAC versions
- **Storage Recovery**: Identify redundant files across folders
- **Library Audit**: Generate JSON inventory of duplicates

## Technical Details

### Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| MP3 | .mp3 | Most common lossy format |
| FLAC | .flac | Lossless compression |
| WAV | .wav | Uncompressed audio |
| AAC | .aac, .m4a | Apple/modern lossy |
| OGG | .ogg | Vorbis codec |
| WMA | .wma | Windows Media Audio |
| Opus | .opus | Modern efficient codec |

### Fingerprint Comparison Algorithm

1. Base64 decode fingerprint strings
2. XOR corresponding bytes
3. Count differing bits (Hamming distance)
4. Calculate similarity ratio
5. Apply length penalty for mismatched durations

## License

MIT License - Feel free to use, modify, and distribute.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Changelog

### Version 1.0.0
- Initial release
- Chromaprint fingerprinting integration
- Multi-format support
- JSON export
- Deletion script generation
- Windows/Unix compatibility
