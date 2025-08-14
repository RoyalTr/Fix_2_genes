import numpy as np
import time

start_time = time.time()

INPUT_FILE = 'results_data_per_generation.txt'
ATTEMPT_OUTPUT = 'aver_each_gen_per_attempt.txt'
SIM_OUTPUT = 'aver_each_gen_per_simulation.txt'


def load_data():
    with open(INPUT_FILE, 'r') as f:
        header = f.readline()
        idx = {name: i for i, name in enumerate(header.strip().split(';'))}

        raw_data = {}
        simul_param = {}

        # Columns for N and frequency data
        col_indices = [idx[col] for col in [
            'N', 'freq_A', 'freq_Aa', 'freq_a',
            'freq_B', 'freq_Bb', 'freq_b', 'pan_heteroz', 'pan_homoz'
        ]]

        # Columns for simulation parameters
        simul_indices = [idx[col] for col in [
            'Ni', 'r', 'K', 's_A', 'h_A', 'p_A_i',
            's_B', 'h_B', 'p_B_i', 'attempts'
        ]]

        for line in f:
            parts = line.strip().split(';')
            SimNr = int(parts[idx['SimNr']])
            attempt = int(parts[idx['attempt']])
            Rep = int(parts[idx['Rep']])
            generation = int(parts[idx['generation']])

            key = (SimNr, attempt, Rep)

            all_data = [float(parts[i]) for i in col_indices]
            N_val = all_data[0]
            freq_data = all_data[1:]

            if key not in raw_data:
                raw_data[key] = []
            # Store as (generation, N_val, freq_data)
            raw_data[key].append((generation, N_val, freq_data))

            # Store simulation parameters once per (SimNr, attempt)
            simul_key = (SimNr, attempt)
            if simul_key not in simul_param:
                simul_param[simul_key] = [
                    int(parts[simul_indices[0]]),   # Ni
                    float(parts[simul_indices[1]]), # r
                    int(parts[simul_indices[2]]),   # K
                    float(parts[simul_indices[3]]), # s_A
                    float(parts[simul_indices[4]]), # h_A
                    float(parts[simul_indices[5]]), # p_A_i
                    float(parts[simul_indices[6]]), # s_B
                    float(parts[simul_indices[7]]), # h_B
                    float(parts[simul_indices[8]]), # p_B_i
                    int(parts[simul_indices[9]])    # attempts
                ]
    return raw_data, simul_param


def complete_and_average_by_generation(raw_data, simul_param):
    result_rows = []

    # Group by (SimNr, attempt)
    attempt_groups = {}
    for (SimNr, attempt, Rep), entries in raw_data.items():
        key = (SimNr, attempt)
        if key not in attempt_groups:
            attempt_groups[key] = []
        attempt_groups[key].append((Rep, entries))

    for (SimNr, attempt), rep_list in attempt_groups.items():
        max_gen_attempt = 0
        rep_data = []

        for Rep, entries in rep_list:
            entries.sort(key=lambda x: x[0])
            max_gen_rep = entries[-1][0]
            last_N = entries[-1][1]
            last_vals = np.array(entries[-1][2])

            if max_gen_rep > max_gen_attempt:
                max_gen_attempt = max_gen_rep

            rep_data.append((Rep, entries, max_gen_rep, last_vals, last_N))

        gen_arrays = {gen: [] for gen in range(max_gen_attempt + 1)}
        gen_N_arrays = {gen: [] for gen in range(max_gen_attempt + 1)}

        for Rep, entries, max_gen_rep, last_vals, last_N in rep_data:
            gen_data = {}
            gen_N_data = {}
            for gen, N_val, vals in entries:
                gen_data[gen] = np.array(vals)
                gen_N_data[gen] = N_val

            for gen in range(max_gen_attempt + 1):
                if gen in gen_data:
                    gen_arrays[gen].append(gen_data[gen])
                    gen_N_arrays[gen].append(gen_N_data[gen])
                else:
                    gen_arrays[gen].append(last_vals)
                    gen_N_arrays[gen].append(last_N)

        for gen in range(max_gen_attempt + 1):
            values_array = np.array(gen_arrays[gen])
            avg = np.mean(values_array, axis=0)

            N_array = np.array(gen_N_arrays[gen])
            avg_N = np.mean(N_array)

            meta = [SimNr, attempt] + simul_param[(SimNr, attempt)] + [gen, avg_N]
            result_rows.append(meta + avg.tolist())

    return result_rows


def compute_per_simulation_averages(attempt_rows):
    attempt_data = {}
    meta_map = {}

    for row in attempt_rows:
        SimNr = int(row[0])
        attempt = int(row[1])
        gen = int(row[12])
        N = float(row[13])
        vals = np.array(row[14:])

        key = (SimNr, attempt)
        if key not in attempt_data:
            attempt_data[key] = {'freq': {}, 'N': {}}

        attempt_data[key]['freq'][gen] = vals
        attempt_data[key]['N'][gen] = N

        if SimNr not in meta_map:
            meta_map[SimNr] = row[2:12]

    max_gen_per_sim = {}
    for (SimNr, attempt), data_dict in attempt_data.items():
        max_gen_attempt = max(data_dict['freq'].keys())
        if SimNr not in max_gen_per_sim:
            max_gen_per_sim[SimNr] = max_gen_attempt
        else:
            max_gen_per_sim[SimNr] = max(max_gen_per_sim[SimNr], max_gen_attempt)

    sim_data = {}
    for (SimNr, attempt), data_dict in attempt_data.items():
        freq_dict = data_dict['freq']
        N_dict = data_dict['N']
        max_gen_attempt_avg = max(freq_dict.keys())
        max_gen_sim = max_gen_per_sim[SimNr]

        last_averaged_vals = freq_dict[max_gen_attempt_avg]
        last_averaged_N = N_dict[max_gen_attempt_avg]

        for gen in range(max_gen_sim + 1):
            if gen not in freq_dict:
                freq_dict[gen] = last_averaged_vals
                N_dict[gen] = last_averaged_N

        if SimNr not in sim_data:
            sim_data[SimNr] = {'freq': {}, 'N': {}}

        for gen in range(max_gen_sim + 1):
            if gen not in sim_data[SimNr]['freq']:
                sim_data[SimNr]['freq'][gen] = []
                sim_data[SimNr]['N'][gen] = []
            sim_data[SimNr]['freq'][gen].append(freq_dict[gen])
            sim_data[SimNr]['N'][gen].append(N_dict[gen])

    sim_rows = []
    for SimNr, data_dict in sim_data.items():
        freq_dict = data_dict['freq']
        N_dict = data_dict['N']
        max_gen_sim = max(freq_dict.keys())

        for gen in range(max_gen_sim + 1):
            values_array = np.array(freq_dict[gen])
            avg = np.mean(values_array, axis=0)

            N_array = np.array(N_dict[gen])
            avg_N = np.mean(N_array)

            row = [SimNr] + meta_map[SimNr] + [gen, avg_N] + avg.tolist()
            sim_rows.append(row)

    return sim_rows


def write_attempt_averages(rows):
    header = ('SimNr;attempt;Ni;r;K;s_A;h_A;p_A_i;s_B;h_B;p_B_i;attempts;generation;N;'
              'Ave_freq_A;Ave_freq_Aa;Ave_freq_a;Ave_freq_B;Ave_freq_Bb;Ave_freq_b;Ave_pan_heteroz;Ave_pan_homoz\n')
    with open(ATTEMPT_OUTPUT, 'w') as f:
        f.write(header)
        for row in rows:
            f.write(';'.join(str(x) for x in row) + '\n')


def write_simulation_averages(rows):
    header = ('SimNr;Ni;r;K;s_A;h_A;p_A_i;s_B;h_B;p_B_i;attempts;generation;N;'
              'Ave_freq_A;Ave_freq_Aa;Ave_freq_a;Ave_freq_B;Ave_freq_Bb;Ave_freq_b;Ave_pan_heteroz;Ave_pan_homoz\n')
    with open(SIM_OUTPUT, 'w') as f:
        f.write(header)
        for row in rows:
            f.write(';'.join(str(x) for x in row) + '\n')


def main():
    print("Loading raw data...")
    raw_data, meta1 = load_data()
    print("Averaging per generation per attempt...")
    attempt_averages = complete_and_average_by_generation(raw_data, meta1)
    print("Writing attempt-level averages...")
    write_attempt_averages(attempt_averages)
    print("Averaging per generation per simulation...")
    simulation_averages = compute_per_simulation_averages(attempt_averages)
    print("Writing simulation-level averages...")
    write_simulation_averages(simulation_averages)
    print("Done.")


if __name__ == '__main__':
    main()

end_time = time.time()
execution_time = end_time - start_time
print(f"\nTotal execution time required: {execution_time:.2f} seconds")
