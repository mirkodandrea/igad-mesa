import os

ews_modes = ["no_ews", "bad_ews", "good_ews", "perfect_ews"]
hrp_levels =["hrp_00", "hrp_30", "hrp_60", "hrp_90", "hrp_30_hi", "hrp_60_hi", "hrp_90_hi"]
scenarios = [f'scenario_{i+1}' for i in range(10)]
basic_income_programs = [True, False]
awareness_programs = [True, False]

# Run all parameter combinations
N = 0
for ews_mode in ews_modes:
    for hrp_level in hrp_levels:
        for scenario in scenarios:
            for basic_income_program in basic_income_programs:
                for awareness_program in awareness_programs:
                    # Run bash script to create new job with these parameters
                    os.makedirs(f"results/batch/scenario_{N+1}", exist_ok=True)
                    os.system("sbatch submit_job.sh " + str(ews_mode) + " " + str(hrp_level) + " " + str(scenario) + " " + str(basic_income_program) + " " + str(awareness_program) + " " + str(N))
                    N += 1
