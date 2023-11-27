import mesa
import pandas as pd
from analysis import do_analysis
from model import EWS_MODES, HOUSE_REPAIR_PROGRAMS_LEVELS, IGAD
from utils import SCENARIOS

N_PROCESSES = 25
N_BATCHES = 500

ews_modes = list(EWS_MODES.keys())
hrp_levels = list(HOUSE_REPAIR_PROGRAMS_LEVELS.keys())

params = dict(
    ews_mode=ews_modes,
    hrp_level=hrp_levels,
    
    basic_income_program=[False, True],
    awareness_program=[False, True],
    
    scenario=SCENARIOS,
    village_0=True,
    village_1=True,
    village_2=True,
    village_3=True,
    village_4=True,
    village_5=True,
    village_6=True,    
)

for batch in range(0, N_BATCHES):
    print(f'Batch {batch+1}')
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
    analysis_df.to_csv(f'batch/analysis_{batch+1}.csv')
    

