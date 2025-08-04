import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict

def extract_scores_from_log(log_file_path):
    """Extract temporal consistency scores from a log file"""
    scores = {}
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()
            
        # Find the last occurrence of "id X: Y.YYYY" pattern
        # This corresponds to the final results
        lines = content.split('\n')
        for line in reversed(lines):
            if line.startswith('id ') and ': ' in line:
                match = re.match(r'id (\d+): (nan|[0-9]+\.[0-9]+)', line)
                if match:
                    id_num = int(match.group(1))
                    value = match.group(2)
                    if value == 'nan':
                        scores[id_num] = np.nan
                    else:
                        scores[id_num] = float(value)
                else:
                    break
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
        return {}
    
    return scores

def analyze_all_logs():
    """Analyze all temporal consistency log files"""
    log_dir = "logs"
    results = defaultdict(list)
    
    # Get all log files
    log_files = [f for f in os.listdir(log_dir) if f.endswith('_tc.log')]
    
    for log_file in log_files:
        # Extract model name from filename
        # Format: video{XX}_{model}_tc.log or video{XX}_img_{model}_tc.log
        parts = log_file.replace('_tc.log', '').split('_')
        
        if len(parts) >= 3:
            if parts[1] == 'img':
                # Format: video{XX}_img_{model} -> img_{model}
                model_name = 'img_' + '_'.join(parts[2:])
            else:
                # Format: video{XX}_{model} -> {model}
                model_name = '_'.join(parts[1:])
        else:
            model_name = 'unknown'
        
        log_path = os.path.join(log_dir, log_file)
        scores = extract_scores_from_log(log_path)
        
        if scores:
            # Convert to array with 10 classes (0-9)
            score_array = np.full(10, np.nan)
            for id_num, score in scores.items():
                if id_num < 10:
                    score_array[id_num] = score
            
            results[model_name].append(score_array)
            print(f"{log_file}: {model_name} - {len(scores)} scores extracted")
    
    # Calculate nanmean for each model
    print("\n" + "="*80)
    print("TEMPORAL CONSISTENCY SCORES SUMMARY")
    print("="*80)
    
    summary_data = []
    for model_name, score_arrays in results.items():
        if score_arrays:
            # Stack all videos for this model
            all_scores = np.vstack(score_arrays)
            
            # Calculate nanmean across all videos for each class
            class_means = np.nanmean(all_scores, axis=0)
            overall_mean = np.nanmean(class_means)
            
            summary_data.append({
                'Model': model_name,
                'Overall_Mean': overall_mean,
                'Num_Videos': len(score_arrays)
            })
            
            print(f"\n{model_name} (from {len(score_arrays)} videos):")
            print(f"  Overall Mean: {overall_mean:.4f}")
            print("  Per-class means:")
            for i, mean_score in enumerate(class_means):
                if not np.isnan(mean_score):
                    print(f"    Class {i}: {mean_score:.4f}")
                else:
                    print(f"    Class {i}: nan")
    
    # Sort by overall mean and display summary table
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Overall_Mean', ascending=False)
    
    print("\n" + "="*80)
    print("RANKED SUMMARY TABLE")
    print("="*80)
    print(summary_df.to_string(index=False, float_format='%.4f'))
    
    return summary_df

if __name__ == "__main__":
    summary = analyze_all_logs() 