def process_max_generation(input_file='results_data_per_generation.txt', output_file='Max_generations.txt'):
    from collections import defaultdict

    max_gen_per_group = defaultdict(int)

    with open(input_file, 'r') as f:
        header = f.readline().strip().split(';')
        # Get indexes of relevant columns
        try:
            simnr_idx = header.index('SimNr')
            attempt_idx = header.index('attempt')
            rep_idx = header.index('Rep')
            gen_idx = header.index('generation')
        except ValueError as e:
            raise RuntimeError(f"Missing required column: {e}")

        for line in f:
            if not line.strip():
                continue  # skip empty lines
            fields = line.strip().split(';')
            try:
                simnr = fields[simnr_idx]
                attempt = fields[attempt_idx]
                rep = fields[rep_idx]
                generation = int(fields[gen_idx])
            except (IndexError, ValueError):
                continue  # skip malformed lines

            key = (simnr, attempt, rep)
            if generation > max_gen_per_group[key]:
                max_gen_per_group[key] = generation

    with open(output_file, 'w') as f:
        f.write('SimNr;attempt;Rep;generation\n')
        for (simnr, attempt, rep), max_gen in max_gen_per_group.items():
            f.write(f'{simnr};{attempt};{rep};{max_gen}\n')

if __name__ == '__main__':
    process_max_generation()

print("Results stored in file Max_generations.txt.")
