import json

def update_paths():
    """
    Remove the absolute path prefix from the audio mapping JSON file.
    """
    # Read the current mapping file
    with open('audio_trustworthy_mapping.json', 'r') as f:
        data = json.load(f)
    
    # Path prefix to remove
    prefix_to_remove = '/Users/yuwen.yu/Desktop/trustworty_sppech_generator/'
    
    # Update each mapping
    updated_data = []
    for item in data:
        updated_item = item.copy()
        
        # Update audio_directory
        if item['audio_directory'].startswith(prefix_to_remove):
            updated_item['audio_directory'] = item['audio_directory'][len(prefix_to_remove):]
        
        # Update file_path
        if item['file_path'].startswith(prefix_to_remove):
            updated_item['file_path'] = item['file_path'][len(prefix_to_remove):]
        
        updated_data.append(updated_item)
    
    # Write the updated data back to the file
    with open('audio_trustworthy_mapping.json', 'w') as f:
        json.dump(updated_data, f, indent=2)
    
    print(f"Updated {len(updated_data)} mappings with relative paths")
    print("Removed prefix: '/Users/yuwen.yu/Desktop/trustworty_sppech_generator/'")

if __name__ == "__main__":
    update_paths() 