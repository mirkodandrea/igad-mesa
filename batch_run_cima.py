import os

import mesa
import pandas as pd
from analysis import do_analysis
from model import EWS_MODES, HOUSE_REPAIR_PROGRAMS_LEVELS, IGAD
from utils import SCENARIOS

N_PROCESSES = 1
N_BATCHES = 1

ews_modes = list(EWS_MODES.keys())
hrp_levels = list(HOUSE_REPAIR_PROGRAMS_LEVELS.keys())

for batch in range(0, N_BATCHES):
    os.makedirs(f'batch/{batch}/', exist_ok=True)
    run = 0
    for hrp_level in hrp_levels[0:1]:
        for basic_income_program in [
            # False,
                True]:
            for awareness_program in [
                # False,
                    True]:

                print(f'Batch {batch+1} - run {run+1}')

                params = dict(
                    ews_mode=ews_modes,
                    hrp_level=hrp_level,

                    basic_income_program=basic_income_program,
                    awareness_program=awareness_program,

                    scenario=SCENARIOS,
                    village_0=True,
                    village_1=True,
                    village_2=True,
                    village_3=True,
                    village_4=True,
                    village_5=True,
                    village_6=True,
                )

                results = mesa.batch_run(
                    IGAD,
                    parameters=params,
                    iterations=1,
                    max_steps=30,
                    number_processes=N_PROCESSES,
                    data_collection_period=1,
                    display_progress=False,
                )
                results_df = pd.DataFrame(results)
                # results_df.to_csv('results.csv')
                analysis_df = do_analysis(results_df)
                analysis_df.to_csv(f'batch/{batch}/analysis_{run+1}.csv')

                run += 1
