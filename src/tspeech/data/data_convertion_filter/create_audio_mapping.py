import pandas as pd
import json
import os
from pathlib import Path

def create_audio_mapping():
    """
    Create a mapping table from CSV files to audio directories with file paths and trustworthy scores.
    """
    
    # Define the mapping between CSV files and their corresponding audio directories
    csv_to_audio_mapping = {
        'processed_df_hum_q.csv': '/Users/yuwen.yu/Desktop/trustworty_sppech_generator/Audio/question_humor_audio',
        'processed_df_rec_hum.csv': '/Users/yuwen.yu/Desktop/trustworty_sppech_generator/Audio/recommendation_humor_audio',
        'processed_df_rec_pol.csv': '/Users/yuwen.yu/Desktop/trustworty_sppech_generator/Audio/recommendation_polite_audio',
        'processed_results_df_rec.csv': '/Users/yuwen.yu/Desktop/trustworty_sppech_generator/Audio/recommendation_netural_audio',
        'processed_results_df_question.csv': '/Users/yuwen.yu/Desktop/trustworty_sppech_generator/Audio/question_neutral_audio',
        'processed_results_df_question_pol_v3.csv': '/Users/yuwen.yu/Desktop/trustworty_sppech_generator/Audio/question_polite_audio'
    }
    
    all_mappings = []
    
    for csv_file, audio_dir in csv_to_audio_mapping.items():
        print(f"Processing {csv_file} -> {audio_dir}")
        
        # Read the CSV file
        csv_path = f"collected_ratings/{csv_file}"
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
        
        # Determine the filename column name based on the CSV structure
        filename_col = None
        for col in ['filename', 'fname']:
            if col in df.columns:
                filename_col = col
                break
        
        if filename_col is None:
            print(f"No filename column found in {csv_file}")
            continue
        
        # Determine the trustworthy column name
        trustworthy_col = None
        for col in ['trustworthy', 'norm_trustworthy', 'trustworthy_x', 'trustworthy_y']:
            if col in df.columns:
                trustworthy_col = col
                break
        
        if trustworthy_col is None:
            print(f"No trustworthy column found in {csv_file}")
            continue
        
        # Process each row
        for idx, row in df.iterrows():
            filename = row[filename_col]
            trustworthy_score = row[trustworthy_col]
            
            # Clean filename (remove .wav extension if present)
            if filename.endswith('.wav'):
                filename = filename[:-4]
            
            # Construct the full file path
            # Look for the file in the audio directory structure
            file_found = False
            
            # Search in the audio directory for matching files
            audio_path = Path(audio_dir)
            if audio_path.exists():
                # Look for files with the same base name
                for audio_file in audio_path.rglob(f"*{filename}*.wav"):
                    full_path = str(audio_file.absolute())
                    all_mappings.append({
                        "csv_file": csv_file,
                        "audio_directory": audio_dir,
                        "file_path": full_path,
                        "trustworthy_score": float(trustworthy_score),
                        "base_filename": filename
                    })
                    file_found = True
                    break
                
                # If not found with .wav extension, try without
                if not file_found:
                    for audio_file in audio_path.rglob(f"*{filename}*"):
                        if audio_file.is_file() and audio_file.suffix.lower() in ['.wav', '.mp3', '.flac']:
                            full_path = str(audio_file.absolute())
                            all_mappings.append({
                                "csv_file": csv_file,
                                "audio_directory": audio_dir,
                                "file_path": full_path,
                                "trustworthy_score": float(trustworthy_score),
                                "base_filename": filename
                            })
                            file_found = True
                            break
            
            if not file_found:
                print(f"Warning: Could not find audio file for {filename} in {audio_dir}")
    
    # Save the mapping to JSON file
    output_file = "audio_trustworthy_mapping.json"
    with open(output_file, 'w') as f:
        json.dump(all_mappings, f, indent=2)
    
    print(f"\nMapping created successfully!")
    print(f"Total mappings: {len(all_mappings)}")
    print(f"Output saved to: {output_file}")
    
    # Print summary statistics
    csv_counts = {}
    for mapping in all_mappings:
        csv_file = mapping['csv_file']
        csv_counts[csv_file] = csv_counts.get(csv_file, 0) + 1
    
    print("\nSummary by CSV file:")
    for csv_file, count in csv_counts.items():
        print(f"  {csv_file}: {count} mappings")
    
    return all_mappings

if __name__ == "__main__":
    create_audio_mapping() 