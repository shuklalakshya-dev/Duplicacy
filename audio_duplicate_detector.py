#!/usr/bin/env python3
"""
Audio Duplicate Detection System
=================================
Identifies duplicate audio files using acoustic fingerprinting technology.
Compares actual sound content regardless of file format, bitrate, or filename.

Author: Audio Duplicate Detector
Version: 1.1.0
"""

import subprocess
import json
import hashlib
import os
import sys
import argparse
import shutil
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set
from datetime import datetime

# Supported audio formats
SUPPORTED_FORMATS = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma', '.opus'}

# Quality ranking for format preference (higher = better quality potential)
FORMAT_QUALITY_RANK = {
    '.flac': 100,  # Lossless
    '.wav': 95,    # Lossless (uncompressed)
    '.m4a': 70,    # AAC container, variable quality
    '.aac': 70,    # Advanced Audio Coding
    '.opus': 65,   # Modern, efficient codec
    '.ogg': 60,    # Vorbis, decent quality
    '.mp3': 50,    # Common, lossy
    '.wma': 40,    # Windows Media Audio
}


@dataclass
class AudioFile:
    """Represents an audio file with its metadata and fingerprint."""
    path: Path
    size: int
    format: str
    fingerprint: Optional[List[int]] = None  # Raw integer array from Chromaprint
    duration: Optional[float] = None
    file_hash: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'path': str(self.path),
            'size': self.size,
            'format': self.format,
            'fingerprint_length': len(self.fingerprint) if self.fingerprint else 0,
            'duration': self.duration,
            'file_hash': self.file_hash,
            'error': self.error
        }


@dataclass
class DuplicateGroup:
    """Represents a group of duplicate audio files."""
    files: List[AudioFile] = field(default_factory=list)
    recommended_keep: Optional[AudioFile] = None
    potential_savings: int = 0
    similarity: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'files': [f.to_dict() for f in self.files],
            'recommended_keep': self.recommended_keep.to_dict() if self.recommended_keep else None,
            'potential_savings': self.potential_savings,
            'similarity': self.similarity,
            'file_count': len(self.files)
        }


class ChromaprintFingerprinter:
    """Handles audio fingerprinting using Chromaprint's fpcalc tool."""
    
    def __init__(self, fpcalc_path: Optional[str] = None):
        """
        Initialize the fingerprinter.
        
        Args:
            fpcalc_path: Path to fpcalc executable. If None, searches PATH.
        """
        self.fpcalc_path = fpcalc_path or self._find_fpcalc()
        if not self.fpcalc_path:
            raise RuntimeError(
                "fpcalc not found. Please install Chromaprint:\n"
                "  Windows: Download from https://acoustid.org/chromaprint\n"
                "           Extract and place fpcalc.exe in this folder or add to PATH\n"
                "  macOS: brew install chromaprint\n"
                "  Linux: apt install libchromaprint-tools"
            )
        print(f"Using fpcalc: {self.fpcalc_path}")
    
    def _find_fpcalc(self) -> Optional[str]:
        """Find fpcalc executable in PATH or common locations."""
        fpcalc_name = 'fpcalc.exe' if sys.platform == 'win32' else 'fpcalc'
        
        # Check common installation paths FIRST (more reliable)
        common_paths = []
        if sys.platform == 'win32':
            script_dir = Path(__file__).parent if '__file__' in dir() else Path.cwd()
            common_paths = [
                Path.cwd() / 'fpcalc.exe',
                script_dir / 'fpcalc.exe',
                Path(os.environ.get('PROGRAMFILES', 'C:\\Program Files')) / 'Chromaprint' / 'fpcalc.exe',
                Path(os.environ.get('PROGRAMFILES', 'C:\\Program Files')) / 'fpcalc' / 'fpcalc.exe',
                Path(os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)')) / 'Chromaprint' / 'fpcalc.exe',
                Path(os.environ.get('LOCALAPPDATA', '')) / 'Chromaprint' / 'fpcalc.exe',
                Path.home() / 'fpcalc.exe',
                Path.home() / 'Downloads' / 'chromaprint-fpcalc-1.5.1-windows-x86_64' / 'fpcalc.exe',
                Path.home() / 'Downloads' / 'fpcalc.exe',
            ]
        else:
            common_paths = [
                Path('/usr/bin/fpcalc'),
                Path('/usr/local/bin/fpcalc'),
                Path('/opt/homebrew/bin/fpcalc'),
                Path.cwd() / 'fpcalc',
            ]
        
        for path in common_paths:
            if path.exists():
                return str(path)
        
        # Try which/where command
        which_cmd = 'where' if sys.platform == 'win32' else 'which'
        try:
            result = subprocess.run(
                [which_cmd, fpcalc_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split('\n')[0]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return None
    
    def get_fingerprint(self, audio_path: Path, duration_limit: int = 120) -> Tuple[Optional[List[int]], Optional[float]]:
        """
        Generate acoustic fingerprint for an audio file.
        
        Args:
            audio_path: Path to the audio file
            duration_limit: Maximum duration to analyze (seconds)
            
        Returns:
            Tuple of (fingerprint_int_array, duration)
        """
        try:
            # Use -raw to get integer array (better for comparison)
            # Use -json for structured output
            cmd = [
                self.fpcalc_path,
                '-raw',       # Get raw fingerprint as integer array
                '-json',      # JSON output format
                '-length', str(duration_limit),
                str(audio_path)
            ]
            
            creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                creationflags=creation_flags
            )
            
            if result.returncode != 0:
                # Try without -raw flag as fallback
                return self._get_fingerprint_base64(audio_path, duration_limit)
            
            data = json.loads(result.stdout)
            fingerprint = data.get('fingerprint')
            duration = data.get('duration')
            
            # With -raw flag, fingerprint is a list of integers
            if isinstance(fingerprint, list) and len(fingerprint) > 0:
                return fingerprint, duration
            
            return None, None
            
        except subprocess.TimeoutExpired:
            return None, None
        except json.JSONDecodeError:
            return None, None
        except Exception as e:
            return None, None
    
    def _get_fingerprint_base64(self, audio_path: Path, duration_limit: int = 120) -> Tuple[Optional[List[int]], Optional[float]]:
        """Fallback method using base64 fingerprint."""
        try:
            cmd = [
                self.fpcalc_path,
                '-json',
                '-length', str(duration_limit),
                str(audio_path)
            ]
            
            creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                creationflags=creation_flags
            )
            
            if result.returncode != 0:
                return None, None
            
            data = json.loads(result.stdout)
            fingerprint_b64 = data.get('fingerprint')
            duration = data.get('duration')
            
            if fingerprint_b64 and isinstance(fingerprint_b64, str):
                # Convert base64 to integer array
                import base64
                import struct
                
                # Chromaprint base64 uses URL-safe encoding
                padding = 4 - len(fingerprint_b64) % 4
                if padding != 4:
                    fingerprint_b64 += '=' * padding
                
                try:
                    raw_bytes = base64.urlsafe_b64decode(fingerprint_b64)
                    # Each integer is 4 bytes (32-bit unsigned)
                    num_ints = len(raw_bytes) // 4
                    fingerprint = list(struct.unpack(f'<{num_ints}I', raw_bytes[:num_ints*4]))
                    return fingerprint, duration
                except:
                    pass
            
            return None, None
            
        except Exception:
            return None, None


class AudioDuplicateDetector:
    """Main class for detecting duplicate audio files."""
    
    def __init__(
        self,
        similarity_threshold: float = 0.95,
        fpcalc_path: Optional[str] = None,
        use_file_hash_fallback: bool = True
    ):
        """
        Initialize the detector.
        
        Args:
            similarity_threshold: Minimum similarity to consider files duplicates (0.0-1.0)
            fpcalc_path: Path to fpcalc executable
            use_file_hash_fallback: Use file hash comparison if fingerprinting fails
        """
        self.similarity_threshold = similarity_threshold
        self.use_file_hash_fallback = use_file_hash_fallback
        
        try:
            self.fingerprinter = ChromaprintFingerprinter(fpcalc_path)
            self.fingerprinting_available = True
        except RuntimeError as e:
            print(f"\nWarning: {e}")
            print("Falling back to file hash comparison only.")
            print("Note: File hash can only detect EXACT duplicates, not same audio in different formats.\n")
            self.fingerprinter = None
            self.fingerprinting_available = False
    
    def scan_folder(self, folder_path: Path, recursive: bool = True) -> List[AudioFile]:
        """
        Scan a folder for audio files.
        
        Args:
            folder_path: Path to the folder to scan
            recursive: Whether to scan subfolders
            
        Returns:
            List of AudioFile objects
        """
        audio_files = []
        pattern = '**/*' if recursive else '*'
        
        print(f"\n{'='*60}")
        print(f"Scanning: {folder_path}")
        print(f"Recursive: {recursive}")
        print(f"Fingerprinting: {'Available' if self.fingerprinting_available else 'NOT AVAILABLE (hash only)'}")
        print(f"{'='*60}\n")
        
        # Collect all audio files
        all_files = list(folder_path.glob(pattern))
        audio_paths = [
            f for f in all_files 
            if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
        ]
        
        print(f"Found {len(audio_paths)} audio files\n")
        
        for i, file_path in enumerate(audio_paths, 1):
            try:
                size = file_path.stat().st_size
                audio_file = AudioFile(
                    path=file_path,
                    size=size,
                    format=file_path.suffix.lower()
                )
                
                # Progress indicator
                progress = f"[{i}/{len(audio_paths)}]"
                print(f"{progress} Processing: {file_path.name}", end='')
                sys.stdout.flush()
                
                # Generate fingerprint
                if self.fingerprinter:
                    fp, duration = self.fingerprinter.get_fingerprint(file_path)
                    if fp and len(fp) > 10:  # Valid fingerprint should have many values
                        audio_file.fingerprint = fp
                        audio_file.duration = duration
                        print(f" [fingerprinted: {len(fp)} samples, {duration:.1f}s]")
                    else:
                        # Fallback to file hash
                        if self.use_file_hash_fallback:
                            audio_file.file_hash = self._compute_file_hash(file_path)
                            print(" [hashed - fingerprint failed]")
                        else:
                            audio_file.error = "Fingerprinting failed"
                            print(" [skipped - fingerprint failed]")
                else:
                    # Only file hash available
                    audio_file.file_hash = self._compute_file_hash(file_path)
                    print(" [hashed]")
                
                audio_files.append(audio_file)
                
            except PermissionError:
                print(f" [error: permission denied]")
            except Exception as e:
                print(f" [error: {str(e)[:50]}]")
        
        return audio_files
    
    def _compute_file_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _popcnt(self, x: int) -> int:
        """Count number of set bits in an integer (population count)."""
        count = 0
        while x:
            count += x & 1
            x >>= 1
        return count
    
    def _calculate_fingerprint_similarity(self, fp1: List[int], fp2: List[int]) -> float:
        """
        Calculate similarity between two fingerprints using bit-error-rate.
        
        Uses Hamming distance on the raw Chromaprint integer arrays.
        This is the standard way to compare Chromaprint fingerprints.
        """
        if not fp1 or not fp2:
            return 0.0
        
        # Use the overlapping portion
        min_len = min(len(fp1), len(fp2))
        max_len = max(len(fp1), len(fp2))
        
        if min_len < 10:  # Too short to compare reliably
            return 0.0
        
        # Calculate bit errors using XOR and popcount
        # Each integer is 32 bits
        total_bits = min_len * 32
        error_bits = 0
        
        for i in range(min_len):
            # XOR gives us differing bits
            xor = fp1[i] ^ fp2[i]
            # Count the differing bits
            error_bits += self._popcnt(xor & 0xFFFFFFFF)
        
        # Calculate similarity (1.0 = identical, 0.0 = completely different)
        similarity = 1.0 - (error_bits / total_bits)
        
        # Apply penalty for very different lengths (different duration songs)
        length_ratio = min_len / max_len
        if length_ratio < 0.5:
            # Very different lengths - likely not duplicates
            similarity *= length_ratio
        elif length_ratio < 0.8:
            # Somewhat different lengths - apply small penalty
            similarity *= (0.8 + 0.2 * length_ratio)
        
        return max(0.0, min(1.0, similarity))
    
    def _calculate_duration_similarity(self, dur1: Optional[float], dur2: Optional[float]) -> float:
        """Calculate duration similarity as a ratio."""
        if not dur1 or not dur2:
            return 1.0  # Don't penalize if duration unknown
        
        # Allow up to 2 seconds difference for same duration
        diff = abs(dur1 - dur2)
        if diff <= 2.0:
            return 1.0
        
        ratio = min(dur1, dur2) / max(dur1, dur2)
        return ratio
    
    def find_duplicates(self, audio_files: List[AudioFile]) -> List[DuplicateGroup]:
        """
        Find duplicate groups among audio files.
        
        Args:
            audio_files: List of AudioFile objects to compare
            
        Returns:
            List of DuplicateGroup objects
        """
        print(f"\n{'='*60}")
        print("Finding duplicates...")
        print(f"Similarity threshold: {self.similarity_threshold}")
        print(f"{'='*60}\n")
        
        # Separate files by comparison method
        fingerprinted = [f for f in audio_files if f.fingerprint]
        hash_only = [f for f in audio_files if f.file_hash and not f.fingerprint]
        
        print(f"Files with fingerprints: {len(fingerprinted)}")
        print(f"Files with hash only: {len(hash_only)}")
        
        duplicate_groups: List[DuplicateGroup] = []
        processed: Set[str] = set()
        
        # Compare fingerprinted files
        if fingerprinted:
            print(f"\nComparing {len(fingerprinted)} fingerprinted files...")
            
            total_comparisons = len(fingerprinted) * (len(fingerprinted) - 1) // 2
            comparison_count = 0
            last_percent = -1
            matches_found = 0
            
            for i, file1 in enumerate(fingerprinted):
                if str(file1.path) in processed:
                    continue
                
                group_files = [file1]
                group_similarities = []
                
                for j, file2 in enumerate(fingerprinted[i+1:], i+1):
                    if str(file2.path) in processed:
                        continue
                    
                    comparison_count += 1
                    
                    # Progress for large libraries
                    if total_comparisons > 100:
                        percent = int(comparison_count / total_comparisons * 100)
                        if percent != last_percent and percent % 10 == 0:
                            print(f"  Progress: {percent}% ({matches_found} matches found)", end='\r')
                            sys.stdout.flush()
                            last_percent = percent
                    
                    # Quick duration check first (filter obvious non-matches)
                    dur_sim = self._calculate_duration_similarity(file1.duration, file2.duration)
                    if dur_sim < 0.85:  # Duration differs by more than 15%
                        continue
                    
                    # Compare fingerprints (the key comparison)
                    fp_sim = self._calculate_fingerprint_similarity(
                        file1.fingerprint, file2.fingerprint
                    )
                    
                    if fp_sim >= self.similarity_threshold:
                        group_files.append(file2)
                        group_similarities.append(fp_sim)
                        processed.add(str(file2.path))
                        matches_found += 1
                
                if len(group_files) > 1:
                    processed.add(str(file1.path))
                    avg_similarity = sum(group_similarities) / len(group_similarities) if group_similarities else 1.0
                    group = DuplicateGroup(files=group_files, similarity=avg_similarity)
                    self._recommend_keep(group)
                    duplicate_groups.append(group)
            
            print(f"\n  Fingerprint comparison complete. Found {len(duplicate_groups)} duplicate groups.")
        
        # Compare hash-only files (exact matches)
        if hash_only:
            print(f"\nComparing {len(hash_only)} hash-only files...")
            
            hash_groups: Dict[str, List[AudioFile]] = defaultdict(list)
            for f in hash_only:
                hash_groups[f.file_hash].append(f)
            
            hash_duplicates = 0
            for hash_val, files in hash_groups.items():
                if len(files) > 1:
                    group = DuplicateGroup(files=files, similarity=1.0)
                    self._recommend_keep(group)
                    duplicate_groups.append(group)
                    hash_duplicates += 1
            
            print(f"  Found {hash_duplicates} exact duplicate groups (by hash)")
        
        # Sort groups by potential savings
        duplicate_groups.sort(key=lambda g: g.potential_savings, reverse=True)
        
        return duplicate_groups
    
    def _recommend_keep(self, group: DuplicateGroup):
        """
        Determine which file to recommend keeping in a duplicate group.
        
        Priority:
        1. Lossless formats (FLAC, WAV)
        2. Larger file size (usually higher quality)
        3. Higher quality format ranking
        """
        files = group.files
        
        # Score each file
        scored_files = []
        for f in files:
            format_score = FORMAT_QUALITY_RANK.get(f.format, 30)
            size_score = f.size / (1024 * 1024)  # MB, normalized
            
            # Bonus for lossless
            lossless_bonus = 50 if f.format in ['.flac', '.wav'] else 0
            
            total_score = format_score + (size_score * 0.1) + lossless_bonus
            scored_files.append((f, total_score))
        
        # Sort by score (highest first)
        scored_files.sort(key=lambda x: x[1], reverse=True)
        
        group.recommended_keep = scored_files[0][0]
        
        # Calculate potential savings (sum of all but the kept file)
        group.potential_savings = sum(
            f.size for f in files if f != group.recommended_keep
        )


class ReportGenerator:
    """Generates reports and deletion scripts."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def display_results(self, groups: List[DuplicateGroup], total_files: int):
        """Display results to console."""
        print(f"\n{'='*60}")
        print("DUPLICATE DETECTION RESULTS")
        print(f"{'='*60}\n")
        
        if not groups:
            print("No duplicate files found!")
            print("\nPossible reasons:")
            print("  - No actual duplicates exist in the folder")
            print("  - Try lowering the threshold (e.g., --threshold 0.90)")
            print("  - If using hash-only mode, only exact copies are detected")
            return
        
        total_duplicates = sum(len(g.files) for g in groups)
        total_savings = sum(g.potential_savings for g in groups)
        
        print(f"Total files scanned: {total_files}")
        print(f"Duplicate groups found: {len(groups)}")
        print(f"Total duplicate files: {total_duplicates}")
        print(f"Potential space savings: {self._format_size(total_savings)}")
        print()
        
        for i, group in enumerate(groups, 1):
            print(f"\n{'-'*50}")
            print(f"DUPLICATE GROUP {i} ({len(group.files)} files)")
            print(f"Similarity: {group.similarity*100:.1f}%")
            print(f"Potential savings: {self._format_size(group.potential_savings)}")
            print(f"{'-'*50}")
            
            for f in group.files:
                status = "KEEP" if f == group.recommended_keep else "DELETE"
                status_color = f"[{status}]"
                print(f"  {status_color:10} {f.path}")
                print(f"             Size: {self._format_size(f.size)}, Format: {f.format.upper()}")
                if f.duration:
                    print(f"             Duration: {f.duration:.1f}s")
    
    def _format_size(self, size_bytes: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} PB"
    
    def export_json(self, groups: List[DuplicateGroup], audio_files: List[AudioFile]) -> Path:
        """Export results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"duplicate_report_{timestamp}.json"
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_files_scanned': len(audio_files),
            'duplicate_groups_found': len(groups),
            'total_duplicate_files': sum(len(g.files) for g in groups),
            'potential_savings_bytes': sum(g.potential_savings for g in groups),
            'potential_savings_human': self._format_size(sum(g.potential_savings for g in groups)),
            'duplicate_groups': [g.to_dict() for g in groups]
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\nJSON report saved: {json_path}")
        return json_path
    
    def generate_deletion_script(
        self, 
        groups: List[DuplicateGroup], 
        platform: str = 'auto'
    ) -> Tuple[Path, Path]:
        """
        Generate safe deletion scripts for user review.
        
        Args:
            groups: List of duplicate groups
            platform: 'windows', 'unix', or 'auto'
            
        Returns:
            Tuple of (windows_script_path, unix_script_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine platform
        if platform == 'auto':
            is_windows = sys.platform == 'win32'
        else:
            is_windows = platform == 'windows'
        
        scripts = []
        
        # Windows batch script
        bat_path = self.output_dir / f"delete_duplicates_{timestamp}.bat"
        bat_content = self._generate_windows_script(groups)
        with open(bat_path, 'w', encoding='utf-8') as f:
            f.write(bat_content)
        scripts.append(bat_path)
        
        # Unix shell script
        sh_path = self.output_dir / f"delete_duplicates_{timestamp}.sh"
        sh_content = self._generate_unix_script(groups)
        with open(sh_path, 'w', encoding='utf-8') as f:
            f.write(sh_content)
        scripts.append(sh_path)
        
        print(f"\nDeletion scripts generated:")
        print(f"  Windows: {bat_path}")
        print(f"  Unix/Mac: {sh_path}")
        print(f"\n⚠️  WARNING: Review these scripts carefully before running!")
        print(f"    Files marked for deletion CANNOT be recovered after execution.")
        print(f"    Edit the script to uncomment (remove REM or #) lines you want to delete.")
        
        return bat_path, sh_path
    
    def _generate_windows_script(self, groups: List[DuplicateGroup]) -> str:
        """Generate Windows batch deletion script."""
        lines = [
            "@echo off",
            "REM =====================================================",
            "REM Audio Duplicate Deletion Script",
            f"REM Generated: {datetime.now().isoformat()}",
            "REM =====================================================",
            "REM",
            "REM WARNING: This script will PERMANENTLY DELETE files!",
            "REM Review each entry carefully before running.",
            "REM",
            "REM To use: Remove 'REM' from lines you want to execute",
            "REM =====================================================",
            "",
            "echo.",
            'echo ========================================',
            'echo  Audio Duplicate Deletion Script',
            'echo ========================================',
            "echo.",
            'echo This script will delete duplicate audio files.',
            'echo Press Ctrl+C to cancel, or',
            "pause",
            ""
        ]
        
        for i, group in enumerate(groups, 1):
            lines.append(f"REM --- Group {i} (Similarity: {group.similarity*100:.1f}%) ---")
            lines.append(f"REM KEEPING: {group.recommended_keep.path}")
            
            for f in group.files:
                if f != group.recommended_keep:
                    # Escape path for batch
                    escaped_path = str(f.path).replace('%', '%%')
                    lines.append(f'REM del "{escaped_path}"')
            
            lines.append("")
        
        lines.extend([
            "echo.",
            "echo Deletion complete!",
            "pause"
        ])
        
        return '\n'.join(lines)
    
    def _generate_unix_script(self, groups: List[DuplicateGroup]) -> str:
        """Generate Unix/macOS shell deletion script."""
        lines = [
            "#!/bin/bash",
            "# =====================================================",
            "# Audio Duplicate Deletion Script",
            f"# Generated: {datetime.now().isoformat()}",
            "# =====================================================",
            "#",
            "# WARNING: This script will PERMANENTLY DELETE files!",
            "# Review each entry carefully before running.",
            "#",
            "# To use: Remove '#' from lines you want to execute",
            "# =====================================================",
            "",
            'echo ""',
            'echo "========================================"',
            'echo " Audio Duplicate Deletion Script"',
            'echo "========================================"',
            'echo ""',
            'echo "This script will delete duplicate audio files."',
            'echo "Press Ctrl+C to cancel, or Enter to continue..."',
            "read",
            ""
        ]
        
        for i, group in enumerate(groups, 1):
            lines.append(f"# --- Group {i} (Similarity: {group.similarity*100:.1f}%) ---")
            lines.append(f"# KEEPING: {group.recommended_keep.path}")
            
            for f in group.files:
                if f != group.recommended_keep:
                    # Escape path for shell
                    escaped_path = str(f.path).replace("'", "'\\''")
                    lines.append(f"# rm '{escaped_path}'")
            
            lines.append("")
        
        lines.extend([
            'echo ""',
            'echo "Deletion complete!"'
        ])
        
        return '\n'.join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Audio Duplicate Detection System - Find duplicate audio files using acoustic fingerprinting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "C:\\Music"
  %(prog)s "C:\\Music" --threshold 0.90
  %(prog)s "C:\\Music" --no-recursive
  %(prog)s "C:\\Music" --output "C:\\Reports"
        """
    )
    
    parser.add_argument(
        'folder',
        type=str,
        nargs='?',
        help='Path to the folder containing audio files'
    )
    
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=0.95,
        help='Similarity threshold (0.0-1.0, default: 0.95)'
    )
    
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        default=True,
        help='Scan subfolders recursively (default: True)'
    )
    
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not scan subfolders'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output directory for reports (default: current directory)'
    )
    
    parser.add_argument(
        '--fpcalc',
        type=str,
        default=None,
        help='Path to fpcalc executable'
    )
    
    parser.add_argument(
        '--no-script',
        action='store_true',
        help='Skip deletion script generation'
    )
    
    parser.add_argument(
        '-v', '--version',
        action='version',
        version='Audio Duplicate Detector 1.1.0'
    )
    
    args = parser.parse_args()
    
    # Interactive mode if no folder provided
    if not args.folder:
        print("\n" + "="*60)
        print("  AUDIO DUPLICATE DETECTION SYSTEM")
        print("="*60)
        print("\nThis tool identifies duplicate audio files using acoustic")
        print("fingerprinting technology - even across different formats!")
        print()
        
        folder = input("Enter the path to your music folder: ").strip().strip('"\'')
        if not folder:
            print("No folder provided. Exiting.")
            return 1
    else:
        folder = args.folder
    
    folder_path = Path(folder)
    
    if not folder_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        return 1
    
    if not folder_path.is_dir():
        print(f"Error: Not a directory: {folder_path}")
        return 1
    
    # Initialize detector
    try:
        detector = AudioDuplicateDetector(
            similarity_threshold=args.threshold,
            fpcalc_path=args.fpcalc
        )
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return 1
    
    # Scan folder
    recursive = args.recursive and not args.no_recursive
    audio_files = detector.scan_folder(folder_path, recursive=recursive)
    
    if not audio_files:
        print("\nNo audio files found in the specified folder.")
        return 0
    
    # Find duplicates
    duplicate_groups = detector.find_duplicates(audio_files)
    
    # Initialize report generator
    output_dir = Path(args.output) if args.output else Path.cwd()
    reporter = ReportGenerator(output_dir)
    
    # Display results
    reporter.display_results(duplicate_groups, len(audio_files))
    
    if duplicate_groups:
        # Export JSON
        reporter.export_json(duplicate_groups, audio_files)
        
        # Generate deletion script
        if not args.no_script:
            reporter.generate_deletion_script(duplicate_groups)
    
    print("\n" + "="*60)
    print("Scan complete!")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
