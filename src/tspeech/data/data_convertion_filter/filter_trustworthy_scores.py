import json
import numpy as np

def filter_trustworthy_scores():
    """
    Filter the audio mapping JSON file to keep only top 30% and bottom 30% based on trustworthy scores.
    """
    # Read the current mapping file
    with open('audio_trustworthy_mapping.json', 'r') as f:
        data = json.load(f)
    
    print(f"Total items: {len(data)}")
    
    # Extract trustworthy scores
    scores = [item['trustworthy_score'] for item in data]
    scores = np.array(scores)
    
    # Calculate percentiles
    top_30_percentile = np.percentile(scores, 70)  # Top 30% (above 70th percentile)
    bottom_30_percentile = np.percentile(scores, 30)  # Bottom 30% (below 30th percentile)
    
    print(f"Top 30% threshold (70th percentile): {top_30_percentile:.4f}")
    print(f"Bottom 30% threshold (30th percentile): {bottom_30_percentile:.4f}")
    
    # Filter items
    filtered_data = []
    top_count = 0
    bottom_count = 0
    
    for item in data:
        score = item['trustworthy_score']
        if score >= top_30_percentile:
            filtered_data.append(item)
            top_count += 1
        elif score <= bottom_30_percentile:
            filtered_data.append(item)
            bottom_count += 1
    
    print(f"Top 30% items: {top_count}")
    print(f"Bottom 30% items: {bottom_count}")
    print(f"Total filtered items: {len(filtered_data)}")
    
    # Save filtered data
    output_file = 'audio_trustworthy_mapping_filtered.json'
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"Filtered data saved to: {output_file}")
    
    # Create summary statistics
    filtered_scores = [item['trustworthy_score'] for item in filtered_data]
    summary = {
        "total_original_items": len(data),
        "total_filtered_items": len(filtered_data),
        "top_30_percent_count": top_count,
        "bottom_30_percent_count": bottom_count,
        "top_30_percentile_threshold": float(top_30_percentile),
        "bottom_30_percentile_threshold": float(bottom_30_percentile),
        "filtered_scores_statistics": {
            "min": float(np.min(filtered_scores)),
            "max": float(np.max(filtered_scores)),
            "mean": float(np.mean(filtered_scores)),
            "median": float(np.median(filtered_scores)),
            "std": float(np.std(filtered_scores))
        }
    }
    
    # Save summary
    with open('filter_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: filter_summary.json")
    
    return filtered_data, summary

if __name__ == "__main__":
    filter_trustworthy_scores() 