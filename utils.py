import pandas as pd
import rasterio as rio
import numpy as np
import geopandas as gpd

from constants import (MATERIAL_STONE_BRICKS, MATERIAL_CONCRETE, MATERIAL_WOOD, MATERIAL_MUD_BRICKS, MATERIAL_INFORMAL_SETTLEMENTS)

MAPS_BASENAME = 'IGAD/Maps/SD_30mHazardMap'

DF_EVENTS = pd.read_csv('IGAD/SD_EventCalendar.csv').query('ReturnPeriod < 273')
SCENARIOS = ['Low Hazard', 'Medium Hazard', 'High Hazard', 'Very High Hazard', 'Extreme Hazard']
MAX_YEARS = 30
# read curves from file, first columns is the index, second column is the value
CURVES = {
    'M': pd.read_csv('IGAD/curves/M1.csv', index_col=0, header=None, names=['damage', 'std']),
    'C': pd.read_csv('IGAD/curves/C1.csv', index_col=0, header=None, names=['damage', 'std']),
    'W': pd.read_csv('IGAD/curves/W1.csv', index_col=0, header=None, names=['damage', 'std']),
    'T': pd.read_csv('IGAD/curves/T1.csv', index_col=0, header=None, names=['damage', 'std']),
    'R': pd.read_csv('IGAD/curves/R1.csv', index_col=0, header=None, names=['damage', 'std'])
}

def generate_scenarios():
    """
    Generate groups of events of max 50 years, based on the return period and the number of events
    Returns a dataframe of 5 event groups with the following columns:
    start_year, end_year, max_return_period, sum_return_period, n_events
    the event groups are defined as follows:
    - Low Hazard: lowest ranking group
    - Medium Hazard, High Hazard, Very High Hazard: middle ranking groups
    - Extreme Hazard: maximium ranking group
    """
    # iterate rows
    groups = []
    last_start_year = None
    max_return_period = 0
    sum_return_period = 0
    n_events = 0
    for i in range(DF_EVENTS.shape[0]):
        row = DF_EVENTS.iloc[i]

        year = row['Year']
        if last_start_year is None:
            last_start_year = year
        
        max_return_period = max(max_return_period, row['ReturnPeriod'])
        sum_return_period += row['ReturnPeriod']
        n_events += 1
        if year - last_start_year > MAX_YEARS:
            groups.append((last_start_year, year, max_return_period, sum_return_period, n_events))
            last_start_year = None
            max_return_period = 0
            sum_return_period = 0
            n_events = 0

    df_groups = pd.DataFrame(groups, columns=['start_year', 'end_year', 'max_return_period', 'sum_return_period', 'n_events'])

    # sort df_groups first by max_return_period and then by n_events
    df_groups.sort_values(by=['max_return_period', 'n_events'], ascending=True, inplace=True)

    # take 5 samples from df_groups
    n_groups = df_groups.shape[0]
    n_choices = len(SCENARIOS)
    
    samples = [round(i*n_groups/(n_choices-1)) for i in range(n_choices-1)] + [n_groups-1]
    df_scenarios = df_groups.iloc[samples].copy()
    df_scenarios['scenario'] = SCENARIOS
    df_scenarios.set_index('scenario', inplace=True)

    return df_scenarios



def get_events(start_year, end_year):
    """
    Returns a dictionary of events, where the key is the relative year and the value is a list of events
    """
    events = {}
    df_floods = DF_EVENTS.query(
        'Year >= @start_year and Year <= @end_year')

    for year, group in df_floods.groupby('Year'):
        for idx, row in group.iterrows():
            filename = f'{MAPS_BASENAME}_{row.ReturnPeriod:0>4d}_cut.tif'
            relative_year = year - start_year
            if relative_year not in events:
                events[relative_year] = []

            events[relative_year].append(
                dict(
                    year=row.Year,
                    interarrival_time=row.InterarrivalTime,
                    filename=filename
                ))
    return events


def load_population_data() -> pd.DataFrame:
    df = pd.read_excel('IGAD/population_data.xlsx')
    awareness_columns = [
        'floods_changed_frequency', 
        'floods_changed_intensity', 
        'floods_changed_predictability', 
        'floods_changed_future_frequency',
        'behaviour_affects_flood_impact'

    ]
    df['fear_of_flood'] = df['fear_of_flood'] / 3
    df['income'] = df['income'].apply(lambda x: (x + np.random.random())**1.3)
    # remove U from awareness columns
    df[awareness_columns] = df[awareness_columns].replace('U', np.NaN)
    df['awareness'] = df[awareness_columns].mean(axis=1) / 3
    df['awareness'][df['awareness']<0.3] = 0.3
    df['awareness'].fillna(0.5, inplace=True)
    
    df['obstacles_to_movement'] = (df[['vulnerabilities', 'properties']].sum(axis=1) > 4) | df['household_size'] > 5
    df['walls_materials'] = df['walls_materials'].apply(lambda m: m.strip())

    df =  df[[
        'village',
        'income',
        'vulnerabilities',
        'properties',
        'walls_materials',
        'fear_of_flood',
        'household_size',
        'awareness',
        'obstacles_to_movement',
    ]]
    df.fillna(0, inplace=True)
    return df

def get_damage(value, material):
    """
    Returns the damage value for a given flood value and material
    @param value: flood value
    @param material: material
    """
    if value <= 0:
        return 0
    
    curve = None
    if material == MATERIAL_STONE_BRICKS:
        curve = CURVES['M']
    elif material == MATERIAL_CONCRETE:
        curve = CURVES['C']
    elif material == MATERIAL_WOOD:
        curve = CURVES['W']
    elif material == MATERIAL_INFORMAL_SETTLEMENTS:
        curve = CURVES['R']
    elif material == MATERIAL_MUD_BRICKS:
        curve = CURVES['T']

    if value < curve.index[0]:
       return 0

    idx = np.searchsorted(curve.index, value, side='left')
    if idx == 0:
        return 0
    elif idx == len(curve.index):
        return curve.iloc[-1]['damage']
    else:        
        prev_value, prev_damage = curve.index[idx-1], curve.iloc[idx-1]['damage']
        next_value, next_damage = curve.index[idx], curve.iloc[idx]['damage']
        damage = prev_damage + (next_damage - prev_damage) * (value - prev_value) / (next_value - prev_value)

    return damage

DF_SCENARIOS = generate_scenarios()
#print(DF_SCENARIOS)


ALL_SETTLEMENTS = gpd.read_file('IGAD/settlements_grid_wdst_sampled.gpkg').to_crs(epsg=4326)
BOUNDING_BOXES = gpd.read_file('IGAD/BoundingBox20022023/BoundingBox_20022023.shp').to_crs(epsg=4326)
# select only the bounding box of the village
ALL_POPULATION_DATA = load_population_data()
