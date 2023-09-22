#%%

import time

import mesa
import numpy as np
import pandas as pd

from model import EWS_MODES, HOUSE_REPAIR_PROGRAMS_LEVELS, IGAD
from utils import SCENARIOS

from convert_batch import process_batch
#%%
# record start time
start_time = time.time()
#%%
sensitivity_params = pd.read_excel('sensitivity/X.xlsx', index_col=0)
sensitivity_params = sensitivity_params.rename(columns={
    'RP': 'RISK_PERCEPTION_THRESHOLD',
    'Low_D': 'LOW_DAMAGE_THRESHOLD',
    'High_D': 'HIGH_DAMAGE_THRESHOLD',
    'Trust': 'TRUST_CHANGE',
    'Fear': 'FEAR_CHANGE',
    'Awareness_inc': 'AWARENESS_INCREASE',
    'Awareness_dec': 'AWARENESS_DECREASE',
    'Fix_D_Neighbours': 'FIX_DAMAGE_NEIGHBOURS',
    'Fix_D_Conc': 'FIX_DAMAGE_CONCRETE',
    'Fix_D_Mud': 'FIX_DAMAGE_MUDBRICK',
    'Informal_settlement': 'FIX_DAMAGE_INFORMAL_SETTLEMENT',
})

model_parameters = [r.to_dict() for idx, r in sensitivity_params.iterrows()]

#%%

ews_modes = list(EWS_MODES.keys())
hrp_levels = list(HOUSE_REPAIR_PROGRAMS_LEVELS.keys())

from concurrent.futures import ProcessPoolExecutor
num_threads = 2

import sys
batch = int(sys.argv[1])

# for batch in range(4900, 5000):
print(f'Batch {batch+1}')
start_run_time = time.time()
params = dict(
    ews_mode='bad_ews',
    hrp_level='hrp_00',
    basic_income_program=False,
    awareness_program=False,
    scenario=SCENARIOS[9],
    village_0=True,
    village_1=False,
    village_2=False,
    village_3=False,
    village_4=False,
    village_5=False,
    village_6=False,
    model_parameters=model_parameters
)
results = mesa.batch_run(
    IGAD,
    parameters=params,
    iterations=1,
    max_steps=30,
    number_processes=70,
    data_collection_period=1,
    display_progress=False,
)

end_run_time = time.time()

results_df = pd.DataFrame(results)

results_df = results_df.set_index(['RunId', 'iteration', 'Step', 'AgentID'])
#results_df.to_csv(f'sensitivity/analysis_batch_{batch+1}_raw.csv')
print(f'Batch {batch+1} Run completed in {end_run_time - start_run_time} seconds')
# set multindex: ['RunId', 'iteration', 'Step', 'AgentID']   
with ProcessPoolExecutor(max_workers=num_threads) as executor:
    analysis_df = process_batch(executor, results_df)
    analysis_df.to_csv(f'sensitivity/analysis_batch_{batch+1}.csv')

end_analysis_time = time.time()
print(f'Batch {batch+1} Analysis completed in {end_analysis_time - end_run_time} seconds')

elapsed_time = time.time() - start_run_time
print(f'Batch {batch+1} Elapsed time: {elapsed_time} seconds')
# calculate estimated time left in hours
if batch > 0:        
    estimated_time_left = elapsed_time * (5000 - batch)
    print(f'Estimated time left: {estimated_time_left / 3600} hours')


