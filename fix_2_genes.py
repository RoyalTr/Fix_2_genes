import numpy as np
import os
import sys
import time
import multiprocessing
from collections import defaultdict

# Global variables user can change
Repetitions = 20  # Number of times to rerun each simulation scenario with the same parameter values. Max is hardware dependent.
generations = 100000000  # Maximum number of generations to run each simulation attempt. Prevents endless runs. Set to a small number to view short initial trajectories or for debugging.
document_results_every_generation = True  # New global variable to control per-generation output

# Prevent system sleep on Windows (useful for long overnight runs)
if os.name == 'nt':
    import ctypes
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

start_time = time.time()

def column_headings():
    """
    Defines the expected column headings for the input_data.txt file.
    These are the key parameters the user provides for each simulation scenario.
    """
    return "Ni;r;K;s_A;attempts;h_A;p_A_i;s_B;h_B;p_B_i"

def results_headings():
    """
    Defines the column headings for the detailed results_data.txt file,
    which stores results for each individual simulation repetition.
    """
    return "SimNr;Rep;Ni;r;K;N_A_fix;N_a_fix;N_B_fix;N_b_fix;s_A;s_B;attempts;h_A;h_B;p_A_i;p_B_i;Prob_A_fix;sd_A_fix;Aver_A_gen;sd_A_gen;Prob_a_fix;sd_a_fix;Aver_a_gen;sd_a_gen;Prob_B_fix;sd_B_fix;Aver_B_gen;sd_B_gen;Prob_b_fix;sd_b_fix;Aver_b_gen;sd_b_gen;Total_Gens"

# ! Modified function to include pan_homoz in per-generation results headings
def per_generation_headings():
    """
    Defines the column headings for the results_data_per_generation.txt file,
    which stores per-generation data when document_results_every_generation is True.
    """
    return "SimNr;attempt;Rep;Ni;r;K;s_A;h_A;p_A_i;s_B;h_B;p_B_i;attempts;generation;N;freq_A;freq_Aa;freq_a;freq_B;freq_Bb;freq_b;pan_heteroz;pan_homoz"

output_headings_avg = "SimNr;Reps;Ni;r;K;N_A_fix;N_a_fix;N_B_fix;N_b_fix;s_A;s_B;attempts;h_A;h_B;p_A_i;p_B_i;Prob_A_fix;sd_A_fix;Aver_A_gen;sd_A_gen;Prob_a_fix;sd_a_fix;Aver_a_gen;sd_a_gen;Prob_B_fix;sd_B_fix;Aver_B_gen;sd_B_gen;Prob_b_fix;sd_b_fix;Aver_b_gen;sd_b_gen;Avg_Total_Gens;Avg_Total_N"
results_filename_avg = "results_data_avg.txt"

example_rows = [
    "100;0.01;1000;0.001;1000;0.5;0.05;0.002;0.7;0.1",
    "1000;0.05;20000;0.005;20000;0.2;0.1;0.001;0.3;0.2"
]

input_filename = "input_data.txt"
headings = column_headings()

# --- Input File Handling and Validation ---
# This section ensures the input_data.txt file exists and contains valid parameters.
if not os.path.exists(input_filename):
    # Create the input file with example data if it doesn't exist
    with open(input_filename, "w") as f:
        f.write(headings + "\n")
        for row in example_rows:
            f.write(row + "\n")
    print("Please enter the parameters to run in file input_data.txt (see example data), then rerun the program.")
    sys.exit(0)
    
with open(input_filename, "r") as f:
    lines = [line.strip() for line in f if line.strip()] # Read non-empty lines

if not lines:
    # If the file is empty after reading, recreate it with examples
    with open(input_filename, "w") as f:
        f.write(headings + "\n")
        for row in example_rows:
            f.write(row + "\n")
    print("The input_data.txt file is empty. Example data has been added. Please enter your parameters and rerun the program.")
    sys.exit(0)

if lines[0] != headings:
    # Validate the header row
    print(f"Error: The heading in input_data.txt is incorrect. Expected: '{headings}' but got '{lines[0]}'")
    print("Please correct the heading in input_data.txt to match the expected format.")
    sys.exit(0)

if len(lines) == 1:
    # If only headings are present, add example rows to guide the user
    with open(input_filename, "a") as f:
        for row in example_rows:
            f.write(row + "\n")
    print("Please enter the parameters you want in file input_data.txt (see example data), then rerun the program.")
    sys.exit(0)

valid_data = [] # List to store parsed and validated simulation scenarios
error_found = False # Flag to track if any validation errors occurred

# Iterate through each data line (skipping the header) to parse and validate parameters
for line_num, line in enumerate(lines[1:], start=2): # start=2 for line numbering from 2
    parts = line.split(";")  # Parameters are delimited by ';'
    if len(parts) != 10:
        print(f"Error: Ten parameters must be found in file input_data.txt in line {line_num}. Please correct and rerun.")
        error_found = True
        continue
    try:
        Ni = int(parts[0]) # Initial population size (e.g., a founder group)
        if not (1 <= Ni <= 1000000000):
            print(f"Error: The value of Ni (initial population size) in line {line_num} is wrong. Must be between 1 and 1,000,000,000. Please correct.")
            error_found = True
            continue
    except ValueError:
        print(f"Error: Invalid data for Ni (initial population size) in line {line_num}. Must be an integer. Please correct.")
        error_found = True
        continue
    try:
        r = float(parts[1]) # Population growth rate per generation
        if not (-10 <= r <= 10):
            print(f"Error: The value of r (growth rate / generation) in line {line_num} must be between -10 and 10. Please correct.")
            error_found = True
            continue
    except ValueError:
        print(f"Error: Invalid data for r (growth rate) in line {line_num}. Must be a number. Please correct.")
        error_found = True
        continue
    try:
        K = int(parts[2]) # Carrying capacity: maximum population size
        if not (K >= Ni): # Carrying capacity must be greater than or equal to initial population size
            print(f"Error: The value of K (carrying capacity) in line {line_num} must be greater or equal to initial population size (Ni). Please correct.")
            error_found = True
            continue
    except ValueError:
        print(f"Error: Invalid data for K (carrying capacity) in line {line_num}. Must be an integer. Please correct.")
        error_found = True
        continue
    try:
        s_A = float(parts[3]) # Selection coefficient for gene A
        if not (-2 <= s_A <= 2):
            print(f"Error: The value of s_A (selection coefficient for Gene A) in line {line_num} is wrong. Must be between -2 and 2. Please correct.")
            error_found = True
            continue
    except ValueError:
        print(f"Error: Invalid data for s_A (selection coefficient for Gene A) in line {line_num}. Must be a number. Please correct.")
        error_found = True
        continue
    try:
        attempts = int(parts[4]) # Number of independent simulation attempts for each scenario
        if not (1 <= attempts <= 10000000000):
            print(f"Error: The value of attempts in line {line_num} is wrong. Must be between 1 and 10,000,000,000. Please correct.")
            error_found = True
            continue
    except ValueError:
        print(f"Error: Invalid data for attempts in line {line_num}. Must be an integer. Please correct.")
        error_found = True
        continue
    try:
        h_A = float(parts[5]) # Dominance coefficient for gene A (0=recessive, 0.5=additive, 1=dominant)
        if not (-1 <= h_A <= 1):
            print(f"Error: The value of h_A (dominance coefficient for Gene A) in line {line_num} is wrong. Must be between -1 and 1. Please correct.")
            error_found = True
            continue
    except ValueError:
        print(f"Error: Invalid data for h_A (dominance coefficient for Gene A) in line {line_num}. Must be a number. Please correct.")
        error_found = True
        continue
    try:
        p_A_i = float(parts[6]) # Initial frequency of the advantageous allele A
        if not (0.0 <= p_A_i <= 1.0):
            print(f"Error: The value of p_A_i (initial frequency of allele A) in line {line_num} must be between 0 and 1. Please correct.")
            error_found = True
            continue
    except ValueError:
        print(f"Error: Invalid data for p_A_i (initial frequency of allele A) in line {line_num}. Must be a number. Please correct.")
        error_found = True
        continue
    try:
        s_B = float(parts[7]) # Selection coefficient for gene B
        if not (-2 <= s_B <= 2):
            print(f"Error: The value of s_B (selection coefficient for Gene B) in line {line_num} is wrong. Must be between -2 and 2. Please correct.")
            error_found = True
            continue
    except ValueError:
        print(f"Error: Invalid data for s_B (selection coefficient for Gene B) in line {line_num}. Must be a number. Please correct.")
        error_found = True
        continue
    try:
        h_B = float(parts[8]) # Dominance coefficient for gene B
        if not (-1 <= h_B <= 1):
            print(f"Error: The value of h_B (dominance coefficient for Gene B) in line {line_num} is wrong. Must be between -1 and 1. Please correct.")
            error_found = True
            continue
    except ValueError:
        print(f"Error: Invalid data for h_B (dominance coefficient for Gene B) in line {line_num}. Must be a number. Please correct.")
        error_found = True
        continue
    try:
        p_B_i = float(parts[9]) # Initial frequency of the advantageous allele B
        if not (0.0 <= p_B_i <= 1.0):
            print(f"Error: The value of p_B_i (initial frequency of allele B) in line {line_num} must be between 0 and 1. Please correct.")
            error_found = True
            continue
    except ValueError:
        print(f"Error: Invalid data for p_B_i (initial frequency of allele B) in line {line_num}. Must be a number. Please correct.")
        error_found = True
        continue

    valid_data.append((Ni, r, K, s_A, attempts, h_A, p_A_i, s_B, h_B, p_B_i))  # Store validated scenario parameters

if error_found:
    print("Please correct the errors in input_data.txt and rerun the program.")
    sys.exit(0)

def simulate_population(Ni, r, K, s_A, p_A_i, s_B, p_B_i, total_generations, attempts, h_A, h_B, sim_idx, rep):
    """
    Simulates population and allele frequency dynamics for two unlinked genes (A and B) over multiple attempts.

    Args:
        Ni (int): Initial population size.
        r (float): Intrinsic growth rate.
        K (int): Carrying capacity.
        s_A (float): Selection coefficient for allele A.
        p_A_i (float): Initial frequency of allele A.
        s_B (float): Selection coefficient for allele B.
        p_B_i (float): Initial frequency of allele B.
        total_generations (int): Maximum generations to simulate per attempt.
        attempts (int): Number of independent simulation runs for these parameters.
        h_A (float): Dominance coefficient for allele A.
        h_B (float): Dominance coefficient for allele B.
        sim_idx (int): Simulation number (index from input_data.txt). # ! Added for per-generation output
        rep (int): Repetition number. # ! Added for per-generation output

    Returns:
        tuple: A collection of aggregated statistics across all attempts, including
               fixation probabilities, average fixation times, and final population sizes.
    """
    # ! Initialize list to store per-generation data if enabled
    per_gen_data = [] if document_results_every_generation else None

    # Initialize counters for aggregating results across all simulation attempts
    A_count = 0  # Number of times allele 'A' fixed
    a_count = 0  # Number of times allele 'a' (A's alternative) fixed (meaning A was lost)
    B_count = 0  # Number of times allele 'B' fixed
    b_count = 0  # Number of times allele 'b' (B's alternative) fixed (meaning B was lost)

    sum_A_fix_gens = 0.0          # Sum of generations taken for 'A' to fix
    sum_A_fix_gens_sq = 0.0       # Sum of squared generations for 'A' fixation (for standard deviation)
    sum_a_fix_gens = 0.0          # Sum of generations for 'a' to fix
    sum_a_fix_gens_sq = 0.0       # Sum of squared generations for 'a' fixation
    sum_B_fix_gens = 0.0
    sum_B_fix_gens_sq = 0.0
    sum_b_fix_gens = 0.0
    sum_b_fix_gens_sq = 0.0
    
    sum_N_A_final = 0.0 # Sum of population sizes when allele 'A' first fixed
    sum_N_a_final = 0.0 # Sum of population sizes when allele 'a' first fixed
    sum_N_B_final = 0.0
    sum_N_b_final = 0.0

    total_generations_sum = 0.0 # Sum of generations required for *both* genes (A and B) to fix across attempts
    total_N_sum = 0.0 # Sum of population sizes when both genes fixed (for averaging)
    successful_both_fixations_count = 0 # Counter for attempts where both genes eventually fixed

    # Pre-calculate fitness values for genotypes for Gene A
    fitness_AA = 1.0 + s_A
    fitness_Aa = 1.0 + h_A * s_A
    fitness_aa = 1.0
    
    # Pre-calculate fitness values for genotypes for Gene B
    fitness_BB = 1.0 + s_B
    fitness_Bb = 1.0 + h_B * s_B
    fitness_bb = 1.0
                                    
    # Iterate through each independent simulation attempt
    for i in range(attempts):
        N = Ni  # Reset population to initial size for each new attempt
        
        # Reset current allele frequencies to initial values for each new attempt
        p_A_t = p_A_i
        p_B_t = p_B_i
        
        # Flags to track fixation status of each gene within the current attempt
        A_fixed_this_attempt = False # True if allele A fixes (frequency 1.0)
        a_fixed_this_attempt = False # True if allele 'a' fixes (frequency 0.0 for A)
        B_fixed_this_attempt = False
        b_fixed_this_attempt = False
        
        # Variables to store the generation number when each gene fixed in this attempt
        gen_A_fixed = 0
        gen_a_fixed = 0
        gen_B_fixed = 0
        gen_b_fixed = 0

        # Pre-calculate constant parts of Beverton-Holt equation for performance
        r_plus_1 = 1.0 + r  # (1 + r)
        r_div_K = r / K     # (r / K)

        # Main simulation loop: iterate through generations
        for gen in range(total_generations):
            # ! Record per-generation data at the start of each generation
            if document_results_every_generation:
                # Calculate genotype frequencies for Gene A
                freq_A = p_A_t
                freq_Aa = 2.0 * p_A_t * (1.0 - p_A_t)
                freq_a = 1.0 - p_A_t
                # Calculate genotype frequencies for Gene B
                freq_B = p_B_t
                freq_Bb = 2.0 * p_B_t * (1.0 - p_B_t)
                freq_b = 1.0 - p_B_t
                # Calculate pan heterozygosity
                pan_heteroz = freq_Aa * freq_Bb
                # Calculate pan homozygosity
                pan_homoz = (freq_A ** 2 + freq_a ** 2) * (freq_B ** 2 + freq_b ** 2)
                # Append data for this generation
                per_gen_data.append((
                    sim_idx, i + 1, rep, Ni, r, K, s_A, h_A, p_A_i, s_B, h_B, p_B_i, attempts,
                    gen, N, freq_A, freq_Aa, freq_a, freq_B, freq_Bb, freq_b, pan_heteroz, pan_homoz
                ))

            # --- Check for Fixation/Loss of Gene A before calculations ---
            # This check is performed early to potentially skip calculations for fixed genes.
            if not (A_fixed_this_attempt or a_fixed_this_attempt):
                # A allele is considered fixed if its frequency is 1.0
                # An allele 'a' is considered fixed if A's frequency is 0.0
                if p_A_t == 0.0:
                    a_fixed_this_attempt = True
                    gen_a_fixed = gen # Record generation when 'a' fixed
                elif p_A_t == 1.0:
                    A_fixed_this_attempt = True
                    gen_A_fixed = gen # Record generation when 'A' fixed

            # --- Check for Fixation/Loss of Gene B before calculations ---
            if not (B_fixed_this_attempt or b_fixed_this_attempt):
                if p_B_t == 0.0:
                    b_fixed_this_attempt = True
                    gen_b_fixed = gen
                elif p_B_t == 1.0:
                    B_fixed_this_attempt = True
                    gen_B_fixed = gen

            # --- Check for overall simulation completion ---
            # If both gene A and gene B have reached fixation (or loss), this attempt is complete.
            if (A_fixed_this_attempt or a_fixed_this_attempt) and \
               (B_fixed_this_attempt or b_fixed_this_attempt):
                
                # Accumulate statistics for this successful attempt
                total_generations_sum += gen # Sum total generations taken for both genes to fix
                total_N_sum += N # Sum the population size at the moment both fixed
                successful_both_fixations_count += 1 # Increment counter for successful dual fixations

                # Accumulate gene-specific fixation statistics
                if A_fixed_this_attempt:
                    A_count += 1
                    sum_A_fix_gens += gen_A_fixed
                    sum_A_fix_gens_sq += gen_A_fixed * gen_A_fixed
                    sum_N_A_final += N
                else: # 'a' fixed
                    a_count += 1
                    sum_a_fix_gens += gen_a_fixed
                    sum_a_fix_gens_sq += gen_a_fixed * gen_a_fixed
                    sum_N_a_final += N

                if B_fixed_this_attempt:
                    B_count += 1
                    sum_B_fix_gens += gen_B_fixed
                    sum_B_fix_gens_sq += gen_B_fixed * gen_B_fixed
                    sum_N_B_final += N
                else: # 'b' fixed
                    b_count += 1
                    sum_b_fix_gens += gen_b_fixed
                    sum_b_fix_gens_sq += gen_b_fixed * gen_b_fixed
                    sum_N_b_final += N
                    
                break # Exit the generation loop for this attempt, move to the next 'attempt'

            # --- Population Dynamics (Beverton-Holt model) ---
            # This calculates the population size for the NEXT generation based on current N and K.
            # This is applied for every generation unless 'r' is zero (fixed population size).
            if r != 0:
                N = round(N * r_plus_1 / (1.0 + r_div_K * N))
                if N < 1: # Population cannot go below 1
                    N = 1
                if N > K: # Population cannot exceed carrying capacity
                    N = K
            
            # If population drops to 0 at any point, it's extinct. No further allele dynamics possible.
            if N == 0:
                # If population goes extinct, any un-fixed alleles are effectively lost.
                if not (A_fixed_this_attempt or a_fixed_this_attempt):
                    a_fixed_this_attempt = True # A is lost (effectively 'a' fixed)
                    gen_a_fixed = gen
                if not (B_fixed_this_attempt or b_fixed_this_attempt):
                    b_fixed_this_attempt = True # B is lost (effectively 'b' fixed)
                    gen_b_fixed = gen
                break # Break out of generation loop if population is extinct

            # --- Allele Frequency Update for Gene A (if not fixed) ---
            if not (A_fixed_this_attempt or a_fixed_this_attempt):
                # Calculate genotype frequencies for current generation (Hardy-Weinberg equilibrium)
                freq_AA = p_A_t * p_A_t
                freq_Aa = 2.0 * p_A_t * (1.0 - p_A_t)
                freq_aa = (1.0 - p_A_t) * (1.0 - p_A_t)

                # Calculate mean fitness of the population for gene A
                mean_fitness_A = freq_AA * fitness_AA + freq_Aa * fitness_Aa + freq_aa * fitness_aa
                
                # Calculate the expected frequency of allele A after selection
                # This is the standard population genetics formula for allele frequency change due to selection
                numerator_A = 2.0 * freq_AA * fitness_AA + freq_Aa * fitness_Aa
                fit_A = numerator_A / (2.0 * mean_fitness_A) if mean_fitness_A > 0 else 0.0 # Avoid division by zero

                # Apply genetic drift: Sample new allele counts using binomial distribution
                # The number of 'A' alleles in the next generation is sampled from a binomial distribution
                # with 2*N trials (total alleles) and probability 'fit_A'.
                n_A_alleles = np.random.binomial(2 * N, float(fit_A))
                p_A_t = n_A_alleles / (2 * N) # Update allele A frequency for next generation
            
            # --- Allele Frequency Update for Gene B (if not fixed) ---
            # Assumes both genes are on different chromosomes and segregate independently.
            if not (B_fixed_this_attempt or b_fixed_this_attempt):
                # Calculate genotype frequencies for current generation
                freq_BB = p_B_t * p_B_t
                freq_Bb = 2.0 * p_B_t * (1.0 - p_B_t)
                freq_bb = (1.0 - p_B_t) * (1.0 - p_B_t)

                # Calculate mean fitness of the population for gene B
                mean_fitness_B = freq_BB * fitness_BB + freq_Bb * fitness_Bb + freq_bb * fitness_bb
                
                # Calculate the expected frequency of allele B after selection
                numerator_B = 2.0 * freq_BB * fitness_BB + freq_Bb * fitness_Bb
                fit_B = numerator_B / (2.0 * mean_fitness_B) if mean_fitness_B > 0 else 0.0 # Avoid division by zero

                # Apply genetic drift: Sample new allele counts using binomial distribution
                n_B_alleles = np.random.binomial(2 * N, float(fit_B))
                p_B_t = n_B_alleles / (2 * N) # Update allele B frequency for next generation
        
        # If the simulation reached max_generations without both genes fixing,
        # their stats for this attempt are effectively not counted towards successful_both_fixations_count
        # and won't affect the averages of fixation times.
        if not ((A_fixed_this_attempt or a_fixed_this_attempt) and (B_fixed_this_attempt or b_fixed_this_attempt)):
            pass # No action needed, already handled by not incrementing successful_both_fixations_count

    # --- Calculate and return aggregated statistics after all attempts ---
    
    # Calculate average final population size for A and a fixations
    avg_N_A = sum_N_A_final / A_count if A_count > 0 else np.nan
    avg_N_a = sum_N_a_final / a_count if a_count > 0 else np.nan
    
    # Calculate probability of A and a fixation
    A_fix_prob = A_count / attempts
    a_fix_prob = a_count / attempts
    
    # Calculate standard deviation of fixation probabilities (binomial distribution)
    A_fix_sd = np.sqrt(A_fix_prob * (1.0 - A_fix_prob) / attempts) if attempts > 0 else np.nan
    a_fix_sd = np.sqrt(a_fix_prob * (1.0 - a_fix_prob) / attempts) if attempts > 0 else np.nan
    
    # Calculate average generations to fixation and its standard deviation for A
    if A_count > 0:
        avg_A_fix_gen = sum_A_fix_gens / A_count
        variance_A = (sum_A_fix_gens_sq / A_count) - (avg_A_fix_gen * avg_A_fix_gen)
        std_A_fix_gen = np.sqrt(variance_A) if variance_A > 0 else 0.0
    else:
        avg_A_fix_gen = np.nan
        std_A_fix_gen = np.nan
            
    # Calculate average generations to fixation and its standard deviation for 'a'
    if a_count > 0:
        avg_a_fix_gen = sum_a_fix_gens / a_count
        variance_a = (sum_a_fix_gens_sq / a_count) - (avg_a_fix_gen * avg_a_fix_gen)
        std_a_fix_gen = np.sqrt(variance_a) if variance_a > 0 else 0.0
    else:
        avg_a_fix_gen = np.nan
        std_a_fix_gen = np.nan
    
    # Repeat calculations for Gene B
    avg_N_B = sum_N_B_final / B_count if B_count > 0 else np.nan
    avg_N_b = sum_N_b_final / b_count if b_count > 0 else np.nan
    B_fix_prob = B_count / attempts
    b_fix_prob = b_count / attempts
    B_fix_sd = np.sqrt(B_fix_prob * (1.0 - B_fix_prob) / attempts) if attempts > 0 else np.nan
    b_fix_sd = np.sqrt(b_fix_prob * (1.0 - b_fix_prob) / attempts) if attempts > 0 else np.nan
    
    if B_count > 0:
        avg_B_fix_gen = sum_B_fix_gens / B_count
        variance_B = (sum_B_fix_gens_sq / B_count) - (avg_B_fix_gen * avg_B_fix_gen)
        std_B_fix_gen = np.sqrt(variance_B) if variance_B > 0 else 0.0
    else:
        avg_B_fix_gen = np.nan
        std_B_fix_gen = np.nan
            
    if b_count > 0:
        avg_b_fix_gen = sum_b_fix_gens / b_count
        variance_b = (sum_b_fix_gens_sq / b_count) - (avg_b_fix_gen * avg_b_fix_gen)
        std_b_fix_gen = np.sqrt(variance_b) if variance_b > 0 else 0.0
    else:
        avg_b_fix_gen = np.nan
        std_b_fix_gen = np.nan

    # Average total generations and population size for attempts where both genes fixed
    avg_total_generations = total_generations_sum / successful_both_fixations_count if successful_both_fixations_count > 0 else np.nan
    avg_total_N = total_N_sum / successful_both_fixations_count if successful_both_fixations_count > 0 else np.nan

    return (avg_N_A, avg_N_a, avg_N_B, avg_N_b,
            A_fix_prob, A_fix_sd, avg_A_fix_gen, std_A_fix_gen, 
            a_fix_prob, a_fix_sd, avg_a_fix_gen, std_a_fix_gen,
            B_fix_prob, B_fix_sd, avg_B_fix_gen, std_B_fix_gen,
            b_fix_prob, b_fix_sd, avg_b_fix_gen, std_b_fix_gen,
            avg_total_generations,
            avg_total_N,
            per_gen_data)  # ! Return per-generation data

def worker(job):
    """
    Worker function for multiprocessing pool.
    Unpacks job parameters, runs simulation, and returns results.
    """
    idx, rep, Ni, r, K, s_A, attempts, h_A, p_A_i, s_B, h_B, p_B_i = job
    results = simulate_population(Ni, r, K, s_A, p_A_i, s_B, p_B_i, generations, attempts, h_A, h_B, idx, rep)  # ! Pass idx and rep
    
    # Unpack results from simulate_population
    (avg_N_A, avg_N_a, avg_N_B, avg_N_b, 
     A_fix_prob, A_fix_sd, avg_A_fix_gen, std_A_fix_gen, 
     a_fix_prob, a_fix_sd, avg_a_fix_gen, std_a_fix_gen,
     B_fix_prob, B_fix_sd, avg_B_fix_gen, std_B_fix_gen,
     b_fix_prob, b_fix_sd, avg_b_fix_gen, std_b_fix_gen,
     avg_total_generations, avg_total_N,
     per_gen_data) = results  # ! Include per_gen_data

    # Return all parameters and calculated results for this specific job
    return (idx, rep, Ni, r, K, 
            avg_N_A, avg_N_a, avg_N_B, avg_N_b,
            s_A, s_B, attempts, h_A, h_B, p_A_i, p_B_i,
            A_fix_prob, A_fix_sd, avg_A_fix_gen, std_A_fix_gen,
            a_fix_prob, a_fix_sd, avg_a_fix_gen, std_a_fix_gen,
            B_fix_prob, B_fix_sd, avg_B_fix_gen, std_B_fix_gen,
            b_fix_prob, b_fix_sd, avg_b_fix_gen, std_b_fix_gen,
            avg_total_generations, avg_total_N,
            per_gen_data)  # ! Include per_gen_data

# --- Main execution block for multiprocessing ---
if __name__ == '__main__':
    max_processes = multiprocessing.cpu_count()  # Determine the number of CPU cores available
    print(f"Maximum number of processes supported by this CPU: {max_processes}")
    
    # Create a list of all individual simulation jobs to be executed.
    # Each job corresponds to one repetition of a scenario defined in input_data.txt.
    jobs = []  
    for idx, (Ni, r, K, s_A, attempts, h_A, p_A_i, s_B, h_B, p_B_i) in enumerate(valid_data, start=1):
        for rep in range(1, Repetitions + 1):
            jobs.append((idx, rep, Ni, r, K, s_A, attempts, h_A, p_A_i, s_B, h_B, p_B_i))
    
    # Use a multiprocessing Pool to distribute jobs across CPU cores.
    # `pool.map` applies the `worker` function to each item in `jobs` in parallel.
    # Results are collected in the `results` list in the order of the input `jobs`.
    with multiprocessing.Pool(processes=max_processes) as pool:
        results = pool.map(worker, jobs)
    
    # Sort individual simulation results for consistent processing and output.
    # Sorting by SimNr (x[0]) and then Repetition (x[1]) ensures that all repetitions
    # for a given scenario are grouped together.
    individual_results_sorted = sorted(results, key=lambda x: (x[0], x[1]))
    
    # ! Write per-generation results if enabled
    if document_results_every_generation:
        per_gen_filename = "results_data_per_generation.txt"
        with open(per_gen_filename, "w") as f:
            f.write(per_generation_headings() + "\n")
            # Collect and sort all per-generation data
            all_per_gen_data = []
            for res in individual_results_sorted:
                per_gen_data = res[-1]  # Last element is per_gen_data
                if per_gen_data:
                    all_per_gen_data.extend(per_gen_data)
            # Sort by SimNr, attempt, Rep, generation
            all_per_gen_data_sorted = sorted(all_per_gen_data, key=lambda x: (x[0], x[1], x[2], x[13]))
            for rec in all_per_gen_data_sorted:
                line = (f"{rec[0]};{rec[1]};{rec[2]};{rec[3]};{rec[4]};{rec[5]};"
                        f"{rec[6]};{rec[7]:.8f};{rec[8]:.8f};{rec[9]};{rec[10]:.8f};{rec[11]:.8f};{rec[12]};"
                        f"{rec[13]};{rec[14]};{rec[15]:.8f};{rec[16]:.8f};{rec[17]:.8f};"
                        f"{rec[18]:.8f};{rec[19]:.8f};{rec[20]:.8f};{rec[21]:.8f};{rec[22]:.8f}")
                f.write(line + "\n")
        print(f"Per-generation results stored in file: {per_gen_filename}.")
    
    # Group and average results over all the Repetitions for each simulation scenario.
    grouped_results = defaultdict(list) # Use defaultdict to easily append results to lists
    
    # Group the sorted individual results by their simulation number (scenario ID).
    for res in individual_results_sorted:
        grouped_results[res[0]].append(res)
    
    results_by_param = [] # List to store averaged results for each unique parameter set
    
    # Iterate through each grouped scenario to calculate average statistics across its repetitions.
    for idx, group in grouped_results.items():
        # Extract specific data points from each individual repetition within the group.
        # List comprehensions provide a concise way to collect these values.
        
        # Data for Gene A
        avg_N_A_list = [r[5] for r in group]
        avg_N_a_list = [r[6] for r in group]
        A_fix_prob_list = [r[16] for r in group]
        A_fix_sd_list = [r[17] for r in group]
        A_fix_gens_list = [r[18] for r in group]
        A_fix_sd_gens_list = [r[19] for r in group]
        a_fix_prob_list = [r[20] for r in group]
        a_fix_sd_list = [r[21] for r in group]
        a_fix_gens_list = [r[22] for r in group]
        a_fix_sd_gens_list = [r[23] for r in group]

        # Data for Gene B
        avg_N_B_list = [r[7] for r in group]
        avg_N_b_list = [r[8] for r in group]
        B_fix_prob_list = [r[24] for r in group]
        B_fix_sd_list = [r[25] for r in group]
        B_fix_gens_list = [r[26] for r in group]
        B_fix_sd_gens_list = [r[27] for r in group]
        b_fix_prob_list = [r[28] for r in group]
        b_fix_sd_list = [r[29] for r in group]
        b_fix_gens_list = [r[30] for r in group]
        b_fix_sd_gens_list = [r[31] for r in group]

        # Data for overall simulation (both genes fixed)
        total_generations_list = [r[32] for r in group]
        total_N_list = [r[33] for r in group]
        
        # Calculate averages for Gene A over all repetitions for this scenario
        avg_A_fix_prob = np.nanmean(A_fix_prob_list)
        avg_A_fix_sd = np.nanmean(A_fix_sd_list)
        avg_A_fix_gens_val = np.nanmean(A_fix_gens_list)
        avg_std_A_fix_gens_val = np.nanmean(A_fix_sd_gens_list)
        
        avg_a_fix_prob = np.nanmean(a_fix_prob_list)
        avg_a_fix_sd = np.nanmean(a_fix_sd_list)
        avg_a_fix_gens_val = np.nanmean(a_fix_gens_list)
        avg_std_a_fix_gens_val = np.nanmean(a_fix_sd_gens_list)
        
        # Calculate averages for Gene B over all repetitions for this scenario
        avg_B_fix_prob = np.nanmean(B_fix_prob_list)
        avg_B_fix_sd = np.nanmean(B_fix_sd_list)
        avg_B_fix_gens_val = np.nanmean(B_fix_gens_list)
        avg_std_B_fix_gens_val = np.nanmean(B_fix_sd_gens_list)
        
        avg_b_fix_prob = np.nanmean(b_fix_prob_list)
        avg_b_fix_sd = np.nanmean(b_fix_sd_list)
        avg_b_fix_gens_val = np.nanmean(b_fix_gens_list)
        avg_std_b_fix_gens_val = np.nanmean(b_fix_sd_gens_list)

        # Calculate average total generations and population for this scenario
        avg_total_gens = np.nanmean(total_generations_list)
        avg_total_N = np.nanmean(total_N_list)

        # Extract the common parameters for this scenario (from the first repetition in the group)
        Ni = group[0][2]
        r = group[0][3]
        K = group[0][4]
        avg_N_A_final = np.nanmean(avg_N_A_list) # Average of the final population size when A fixed
        avg_N_a_final = np.nanmean(avg_N_a_list) # Average of the final population size when a fixed
        avg_N_B_final = np.nanmean(avg_N_B_list)
        avg_N_b_final = np.nanmean(avg_N_b_list)
        s_A = group[0][9]
        s_B = group[0][10]
        attempts = group[0][11] # Note: This 'attempts' is the parameter from input_data, not Repetitions
        h_A = group[0][12]
        h_B = group[0][13]
        p_A_i = group[0][14]
        p_B_i = group[0][15]

        # Append the averaged results for this scenario
        results_by_param.append((idx, Ni, r, K, 
                                avg_N_A_final, avg_N_a_final, avg_N_B_final, avg_N_b_final,
                                s_A, s_B, attempts, h_A, h_B, p_A_i, p_B_i,
                                avg_A_fix_prob, avg_A_fix_sd, avg_A_fix_gens_val, avg_std_A_fix_gens_val,
                                avg_a_fix_prob, avg_a_fix_sd, avg_a_fix_gens_val, avg_std_a_fix_gens_val,
                                avg_B_fix_prob, avg_B_fix_sd, avg_B_fix_gens_val, avg_std_B_fix_gens_val,
                                avg_b_fix_prob, avg_b_fix_sd, avg_b_fix_gens_val, avg_std_b_fix_gens_val,
                                avg_total_gens, avg_total_N))
    
    # --- Write individual simulation results to "results_data.txt" ---
    results_filename = "results_data.txt"
    output_headings = results_headings()
    lines_to_write = []

    # Format each individual simulation record into a string for writing
    for rec in individual_results_sorted:
        line = (f"{rec[0]};{rec[1]};{rec[2]};{rec[3]};{rec[4]};"
                f"{rec[5]:.2f};{rec[6]:.2f};{rec[7]:.2f};{rec[8]:.2f};"
                f"{rec[9]};{rec[10]};{rec[11]};{rec[12]:.8f};{rec[13]:.8f};{rec[14]:.8f};{rec[15]:.8f};"
                f"{rec[16]:.8f};{rec[17]:.8f};{rec[18]:.8f};{rec[19]:.8f};"
                f"{rec[20]:.8f};{rec[21]:.8f};{rec[22]:.8f};{rec[23]:.8f};"
                f"{rec[24]:.8f};{rec[25]:.8f};{rec[26]:.8f};{rec[27]:.8f};"
                f"{rec[28]:.8f};{rec[29]:.8f};{rec[30]:.8f};{rec[31]:.8f};"
                f"{rec[32]:.2f}") # Total_Gens is the 32nd element (0-indexed)
        lines_to_write.append(line)
    
    # Write to file: append if exists, create with header if not
    if os.path.exists(results_filename):
        with open(results_filename, "a") as f:
            for line in lines_to_write:
                f.write(line + "\n")
    else:
        with open(results_filename, "w") as f:
            f.write(output_headings + "\n")
            for line in lines_to_write:
                f.write(line + "\n")
    
    print(f"Detailed results for each simulation repetition stored in file: {results_filename}.")
    
    # --- Write averaged simulation results to "results_data_avg.txt" (if Repetitions > 1) ---
    if Repetitions > 1:
        avg_lines = []

        # Format each averaged scenario record into a string for writing
        for rec in results_by_param:
            line = (f"{rec[0]};{Repetitions};{rec[1]};{rec[2]};{rec[3]};"
                                f"{rec[4]:.2f};{rec[5]:.2f};{rec[6]:.2f};{rec[7]:.2f};"
                                f"{rec[8]};{rec[9]};{rec[10]};{rec[11]:.8f};{rec[12]:.8f};{rec[13]:.8f};{rec[14]:.8f};"
                                f"{rec[15]:.8f};{rec[16]:.8f};{rec[17]:.8f};{rec[18]:.8f};"
                                f"{rec[19]:.8f};{rec[20]:.8f};{rec[21]:.8f};{rec[22]:.8f};"
                                f"{rec[23]:.8f};{rec[24]:.8f};{rec[25]:.8f};{rec[26]:.8f};"
                                f"{rec[27]:.8f};{rec[28]:.8f};{rec[29]:.8f};{rec[30]:.8f};"
                                f"{rec[31]:.2f};{rec[32]:.2f}")
            avg_lines.append(line)

        # Check if results_data_avg.txt exists. If not, create it with headings. Otherwise, append to it.
        if not os.path.exists(results_filename_avg):
            with open(results_filename_avg, "w") as f:
                f.write(output_headings_avg + "\n")
        
        with open(results_filename_avg, "a") as f:
            for line in avg_lines:
                f.write(line + "\n")
        print(f"The average values based on the repetitions were stored in file: {results_filename_avg}")
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nTotal execution time required: {execution_time:.2f} seconds")