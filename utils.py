import pandas as pd
import rasterio as rio
import numpy as np
import geopandas as gpd
from typing import List, Union


from constants import (MATERIAL_STONE_BRICKS, MATERIAL_CONCRETE, MATERIAL_WOOD, MATERIAL_MUD_BRICKS, MATERIAL_INFORMAL_SETTLEMENTS)

MAPS_BASENAME = 'IGAD/Maps/SD_30mHazardMap'

DF_EVENTS = pd.read_csv('IGAD/SD_EventCalendar.csv').query('ReturnPeriod < 273')
#SCENARIOS = ['Low Hazard', 'Medium Hazard', 'High Hazard', 'Very High Hazard', 'Extreme Hazard']
SCENARIOS = [f'scenario_{i+1}' for i in range(10)]

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
    df = pd.read_excel('IGAD/population_data_20230830.xlsx', sheet_name='Filled')

    df = df.rename(columns={
        'Village': 'village',
        'House_mat': 'walls_materials',
        'HH_size': 'household_size',
        'HH_income': 'income',
        'Own_house': 'house',
        'Own_crops': 'croplands',
        'Own_cattle': 'livestock',
        'N_floods': 'number_of_floods',
        'Fear_floods': 'fear_of_flood',
    })
    

    awareness_columns = ['PR_FL_freq', 'PR_FL_int', 'PR_FL_pred', 'PR_FL_future', 'PR_riskFlood', 'PR_riskDispl']

    df['fear_of_flood'] = df['fear_of_flood'] / 3
    df['income'] = df['income'].where(df['income'] > 0.5, 0.5)
    df['health_issues'] = df['Chr_illness'] | df['Disability']

    # remove U from awareness columns
    df[awareness_columns] = df[awareness_columns].replace('U', np.NaN)
    df['awareness'] = df[awareness_columns].mean(axis=1) / 3
    df['awareness'] = df['awareness'].where(df['awareness'] > 0.3, 0.3)
    df['awareness'].fillna(0.5, inplace=True)
    df['walls_materials'] = df['walls_materials'].apply(lambda m: m.strip())

    df =  df[[
        'village',
        'income',
        'livestock',
        'house',
        'croplands',
        'number_of_floods',
        'walls_materials',
        'fear_of_flood',
        'household_size',
        'awareness',
        'health_issues'
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

def get_livelihood_damage(
        flood_value: float,
        livelihood_type: Union['crops','livestock','shop']
    ):
    """
    Returns the damage to livelihoods for a given flood value
    @param flood_value: flood value in mm
    @param livelihood_type: livelihood type
    """
    if livelihood_type == 'crops':
        # min damage 10% at 100mm, max damage 100% at 1000m
        min_flood_value = 100
        max_flood_value = 1000
        if flood_value <= min_flood_value:
            return 0
        return 0.1 + 0.9 * (flood_value - min_flood_value) / (max_flood_value - min_flood_value)
    
    elif livelihood_type == 'livestock':
        pass
    elif livelihood_type == 'shop':
        pass
    return 0


class SimulationData(object):
    def __init__(self):
        """
        Load data from population, settlements and flood events.
        """
        self.incomes = []
        self.flood_prones = []
        self.awarenesses = []
        self.house_materials = []
        self.households_size = []
        self.fears = []
        self.positions = []
        self.villages = []

        self.number_of_floods = []
        self.health_issues = []
        self.livestock = []
        self.house = []
        self.cropland = []

        villages = BOUNDING_BOXES['village'].unique()
        for village in villages:

            bounding_boxes = BOUNDING_BOXES.query('village == @village')            
            for id, bounding_box in bounding_boxes.iterrows():
                flood_prone = bounding_box.floodprone == 1
                settlements = ALL_SETTLEMENTS[ALL_SETTLEMENTS.geometry.within(bounding_box.geometry)]

                n_households = len(settlements)

                village_lons = settlements.geometry.centroid.x
                village_lats = settlements.geometry.centroid.y

                village_flood_prones = [flood_prone] * n_households
                village_positions = list(zip(village_lons, village_lats))
                village_data = ALL_POPULATION_DATA\
                        .query('village == @village')\
                        .sample(n_households, replace=True)

                self.positions += village_positions
                self.incomes += village_data['income'].values.tolist()
                self.flood_prones += village_flood_prones #.tolist()
                self.awarenesses += village_data['awareness'].values.tolist()
                self.house_materials += village_data['walls_materials'].values.tolist()
                self.households_size += village_data['household_size'].values.tolist()

                self.fears += village_data['fear_of_flood'].tolist()
                self.number_of_floods += village_data['number_of_floods'].values.tolist()
                self.health_issues += village_data['health_issues'].values.tolist()
                self.livestock += village_data['livestock'].values.tolist()
                self.house += village_data['house'].values.tolist()
                self.cropland += village_data['croplands'].values.tolist()

                self.villages += [village] * n_households



DF_SCENARIOS = generate_scenarios()
#print(DF_SCENARIOS)


ALL_SETTLEMENTS = gpd.read_file('IGAD/settlements_grid_wdst_sampled.gpkg').to_crs(epsg=4326)
BOUNDING_BOXES = gpd.read_file('IGAD/BoundingBox20022023/BoundingBox_20022023.shp').to_crs(epsg=4326)
# select only the bounding box of the village
ALL_POPULATION_DATA = load_population_data()
