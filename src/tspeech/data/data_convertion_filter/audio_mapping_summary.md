# Audio Trustworthy Score Mapping Summary

## Overview
This document summarizes the mapping between CSV files containing trustworthy scores and their corresponding audio directories.

## Mapping Table

| CSV File | Audio Directory | Mappings Found | Description |
|----------|----------------|----------------|-------------|
| `processed_df_hum_q.csv` | `Audio/question_humor_audio` | 168 | Question humor audio files |
| `processed_df_rec_hum.csv` | `Audio/recommendation_humor_audio` | 133 | Recommendation humor audio files |
| `processed_df_rec_pol.csv` | `Audio/recommendation_polite_audio` | 201 | Recommendation polite audio files |
| `processed_results_df_rec.csv` | `Audio/recommendation_netural_audio` | 349 | Recommendation neutral audio files |
| `processed_results_df_question.csv` | `Audio/question_neutral_audio` | 360 | Question neutral audio files |
| `processed_results_df_question_pol_v3.csv` | `Audio/question_polite_audio` | 202 | Question polite audio files |

## Total Statistics
- **Total mappings**: 1,413
- **Output file**: `audio_trustworthy_mapping.json`
- **File format**: JSON with the following structure:
  ```json
  {
    "csv_file": "filename.csv",
    "audio_directory": "Audio/relative/path/to/directory",
    "file_path": "Audio/relative/path/to/audio/file.wav",
    "trustworthy_score": 0.123456789,
    "base_filename": "base_filename_without_extension"
  }
  ```

## Notes
- Some files from the CSV could not be found in the audio directories (warnings were shown during processing)
- The mapping includes both `.wav` files and other audio formats
- Trustworthy scores are preserved as floating-point numbers from the original CSV files
- File paths are now relative paths for better portability across different systems 