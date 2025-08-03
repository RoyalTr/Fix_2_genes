import numpy as np
from numba import njit, types
from numba.typed import Dict, List
import time

start_time = time.time()

INPUT_FILE = 'results_data_per_generation.txt'
ATTEMPT_OUTPUT = 'aver_each_gen_per_attempt.txt'
SIM_OUTPUT = 'aver_each_gen_per_simulation.txt'

@njit
def average_rows_optimized(values_array):
    """Optimized row averaging using vectorized operations where possible"""
    return np.mean(values_array, axis=0)

@njit
def pad_and_average_reps(rep_data_list, max_generations):
    """
    JIT-compiled function to handle Rep-level padding and averaging
    rep_data_list: List of (max_gen, last_vals_array) for each Rep
    max_generations: Maximum generation across all Reps in this (SimNr, attempt)
    """
    n_reps = len(rep_data_list)
    n_features = 8  # freq_A, freq_Aa, freq_a, freq_B, freq_Bb, freq_b, pan_heteroz, pan_homoz
    
    # Pre-allocate result array
    result = np.zeros((max_generations + 1, n_features))
    
    for gen in range(max_generations + 1):
        gen_sum = np.zeros(n_features)
        
        for rep_idx in range(n_reps):
            max_gen_rep, last_vals = rep_data_list[rep_idx]
            
            if gen <= max_gen_rep:
                # Use actual data (we'll pass the generation data separately)
                pass  # This will be handled by the calling function
            else:
                # Use last values for padding
                gen_sum += last_vals
        
        # This function will be called after actual data is added
        result[gen] = gen_sum / n_reps
    
    return result

@njit
def create_generation_lookup(entries):
    """Create a fast lookup array for generation data"""
    max_gen = 0
    for gen, _ in entries:
        if gen > max_gen:
            max_gen = gen
    
    # Initialize with -1 to indicate missing data
    lookup = np.full((max_gen + 1, 8), -999.0)
    
    for gen, vals in entries:
        for i in range(8):
            lookup[gen, i] = vals[i]
    
    return lookup, max_gen

def parse_header_optimized(header_line):
    """Optimized header parsing"""
    headers = header_line.strip().split(';')
    return {name: i for i, name in enumerate(headers)}

def load_data_optimized():
    """Optimized data loading with better memory usage"""
    with open(INPUT_FILE, 'r') as f:
        header = f.readline()
        idx = parse_header_optimized(header)
        
        # Use more efficient data structures
        raw_data = {}
        metadata = {}
        N_map = {}
        
        # Pre-define column indices for faster access
        col_indices = [idx[col] for col in [
            'freq_A', 'freq_Aa', 'freq_a',
            'freq_B', 'freq_Bb', 'freq_b', 'pan_heteroz', 'pan_homoz']]
        
        meta_indices = [idx[col] for col in [
            'Ni', 'r', 'K', 's_A', 'h_A', 'p_A_i', 
            's_B', 'h_B', 'p_B_i', 'attempts']]
        
        for line in f:
            parts = line.strip().split(';')
            SimNr = int(parts[idx['SimNr']])
            attempt = int(parts[idx['attempt']])
            Rep = int(parts[idx['Rep']])
            generation = int(parts[idx['generation']])
            
            key = (SimNr, attempt, Rep)
            
            # Extract frequency data more efficiently
            freq_data = [float(parts[i]) for i in col_indices]
            
            if key not in raw_data:
                raw_data[key] = []
            raw_data[key].append((generation, freq_data))
            
            # Store metadata only once per (SimNr, attempt)
            meta_key = (SimNr, attempt)
            if meta_key not in metadata:
                metadata[meta_key] = [
                    int(parts[meta_indices[0]]),  # Ni
                    float(parts[meta_indices[1]]), # r
                    int(parts[meta_indices[2]]),   # K
                    float(parts[meta_indices[3]]), # s_A
                    float(parts[meta_indices[4]]), # h_A
                    float(parts[meta_indices[5]]), # p_A_i
                    float(parts[meta_indices[6]]), # s_B
                    float(parts[meta_indices[7]]), # h_B
                    float(parts[meta_indices[8]]), # p_B_i
                    int(parts[meta_indices[9]])    # attempts
                ]
            
            N_map[(SimNr, attempt, generation)] = int(parts[idx['N']])
    
    return raw_data, metadata, N_map

def complete_and_average_by_generation_optimized(raw_data, metadata, N_map):
    """Optimized generation completion and averaging"""
    result_rows = []
    
    # Group by (SimNr, attempt) for efficient processing
    attempt_groups = {}
    for (SimNr, attempt, Rep), entries in raw_data.items():
        key = (SimNr, attempt)
        if key not in attempt_groups:
            attempt_groups[key] = []
        attempt_groups[key].append((Rep, entries))
    
    for (SimNr, attempt), rep_list in attempt_groups.items():
        # Find maximum generation across all Reps in this attempt
        max_gen_attempt = 0
        rep_data = []
        
        for Rep, entries in rep_list:
            # Sort entries by generation (only once)
            entries.sort(key=lambda x: x[0])
            max_gen_rep = entries[-1][0]
            last_vals = np.array(entries[-1][1])
            
            if max_gen_rep > max_gen_attempt:
                max_gen_attempt = max_gen_rep
            
            rep_data.append((Rep, entries, max_gen_rep, last_vals))
        
        # Pre-allocate arrays for this attempt
        gen_arrays = {}
        for gen in range(max_gen_attempt + 1):
            gen_arrays[gen] = []
        
        # Process each Rep
        for Rep, entries, max_gen_rep, last_vals in rep_data:
            # Create generation lookup for fast access
            gen_data = {}
            for gen, vals in entries:
                gen_data[gen] = np.array(vals)
            
            # Fill in data for each generation
            for gen in range(max_gen_attempt + 1):
                if gen in gen_data:
                    gen_arrays[gen].append(gen_data[gen])
                else:
                    # Use last values for padding (Step 1 padding)
                    gen_arrays[gen].append(last_vals)
        
        # Average across Reps for each generation
        for gen in range(max_gen_attempt + 1):
            values_array = np.array(gen_arrays[gen])
            avg = np.mean(values_array, axis=0)
            
            # Keep the original averaged heterozygote frequencies
            # avg[1] = Ave_freq_Aa (direct average of freq_Aa across reps)
            # avg[4] = Ave_freq_Bb (direct average of freq_Bb across reps)
            # avg[6] = pan_heteroz (direct average of pan_heteroz across reps)
            # avg[7] = pan_homoz (direct average of pan_homoz across reps)
            
            # Get N value
            N_key = (SimNr, attempt, gen)
            if N_key in N_map:
                N_val = N_map[N_key]
            else:
                # Find max generation for this (SimNr, attempt)
                max_gen = max(g for (s, a, g) in N_map.keys() if s == SimNr and a == attempt)
                N_val = N_map[(SimNr, attempt, max_gen)]
            
            meta = [SimNr, attempt] + metadata[(SimNr, attempt)] + [gen, N_val]
            result_rows.append(meta + avg.tolist())
    
    return result_rows

def compute_per_simulation_averages_optimized(attempt_rows):
    """Optimized simulation-level averaging with correct padding logic"""
    # Group by (SimNr, attempt) first to handle padding correctly
    attempt_data = {}
    meta_map = {}
    N_map = {}
    
    for row in attempt_rows:
        SimNr = int(row[0])
        attempt = int(row[1])
        gen = int(row[12])
        N = int(row[13])
        vals = np.array(row[14:])
        
        key = (SimNr, attempt)
        if key not in attempt_data:
            attempt_data[key] = {}
        
        attempt_data[key][gen] = vals
        N_map[(SimNr, gen)] = N
        
        if SimNr not in meta_map:
            meta_map[SimNr] = row[2:12]
    
    # Step 2 Padding: For each (SimNr, attempt), extend to max_gen_Sim
    sim_data = {}
    
    # First, find max_gen_Sim for each SimNr
    max_gen_per_sim = {}
    for (SimNr, attempt), gen_dict in attempt_data.items():
        max_gen_attempt = max(gen_dict.keys())
        if SimNr not in max_gen_per_sim:
            max_gen_per_sim[SimNr] = max_gen_attempt
        else:
            max_gen_per_sim[SimNr] = max(max_gen_per_sim[SimNr], max_gen_attempt)
    
    # Now pad each (SimNr, attempt) to max_gen_Sim
    for (SimNr, attempt), gen_dict in attempt_data.items():
        max_gen_attempt_avg = max(gen_dict.keys())  # This attempt's max generation
        max_gen_sim = max_gen_per_sim[SimNr]        # This simulation's max generation
        
        # Get the last averaged values for this attempt
        last_averaged_vals = gen_dict[max_gen_attempt_avg]
        
        # Extend this attempt's data to max_gen_Sim using its last_averaged_vals
        for gen in range(max_gen_sim + 1):
            if gen not in gen_dict:
                # Pad with last_averaged_vals (Step 2 padding)
                gen_dict[gen] = last_averaged_vals
        
        # Store extended data for simulation averaging
        if SimNr not in sim_data:
            sim_data[SimNr] = {}
        
        for gen in range(max_gen_sim + 1):
            if gen not in sim_data[SimNr]:
                sim_data[SimNr][gen] = []
            sim_data[SimNr][gen].append(gen_dict[gen])
    
    # Now average across attempts for each (SimNr, generation)
    sim_rows = []
    
    for SimNr, gen_dict in sim_data.items():
        max_gen_sim = max(gen_dict.keys())
        
        for gen in range(max_gen_sim + 1):
            values_array = np.array(gen_dict[gen])
            avg = np.mean(values_array, axis=0)
            
            # Keep the original averaged heterozygote frequencies
            # avg[1] = Ave_freq_Aa (average of Ave_freq_Aa across attempts)
            # avg[4] = Ave_freq_Bb (average of Ave_freq_Bb across attempts)
            # avg[6] = pan_heteroz (average of pan_heteroz across attempts)
            # avg[7] = pan_homoz (average of pan_homoz across attempts)
            
            # Get N value
            max_g = max(g for (s, g) in N_map.keys() if s == SimNr)
            N_val = N_map.get((SimNr, gen), N_map[(SimNr, max_g)])
            
            row = [SimNr] + meta_map[SimNr] + [gen, N_val] + avg.tolist()
            sim_rows.append(row)
    
    return sim_rows

def write_attempt_averages_optimized(rows):
    """Optimized file writing"""
    header = ('SimNr;attempt;Ni;r;K;s_A;h_A;p_A_i;s_B;h_B;p_B_i;attempts;generation;N;'
              'Ave_freq_A;Ave_freq_Aa;Ave_freq_a;Ave_freq_B;Ave_freq_Bb;Ave_freq_b;Ave_pan_heteroz;Ave_pan_homoz\n')
    
    with open(ATTEMPT_OUTPUT, 'w') as f:
        f.write(header)
        # Use join once per row instead of multiple string operations
        for row in rows:
            f.write(';'.join(str(x) for x in row) + '\n')

def write_simulation_averages_optimized(rows):
    """Optimized file writing"""
    header = ('SimNr;Ni;r;K;s_A;h_A;p_A_i;s_B;h_B;p_B_i;attempts;generation;N;'
              'Ave_freq_A;Ave_freq_Aa;Ave_freq_a;Ave_freq_B;Ave_freq_Bb;Ave_freq_b;Ave_pan_heteroz;Ave_pan_homoz\n')
    
    with open(SIM_OUTPUT, 'w') as f:
        f.write(header)
        for row in rows:
            f.write(';'.join(str(x) for x in row) + '\n')

def main():
    print("Loading raw data...")
    raw_data, meta1, N_map = load_data_optimized()
    print("Averaging per generation per attempt...")
    attempt_averages = complete_and_average_by_generation_optimized(raw_data, meta1, N_map)
    print("Writing attempt-level averages...")
    write_attempt_averages_optimized(attempt_averages)
    print("Averaging per generation per simulation...")
    simulation_averages = compute_per_simulation_averages_optimized(attempt_averages)
    print("Writing simulation-level averages...")
    write_simulation_averages_optimized(simulation_averages)
    print("Done.")

if __name__ == '__main__':
    main()

end_time = time.time()
execution_time = end_time - start_time
print(f"\nTotal execution time required: {execution_time:.2f} seconds")