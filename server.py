import geopandas as gpd
import mesa_geo as mg
import numpy as np
import pandas as pd
from numpy.random import normal, pareto, poisson, random

import mesa
from agents import HouseholdAgent
from constants import MAX_YEARS, POVERTY_LINE
from model import IGAD
from utils import get_events, load_population_data


class IGADText(mesa.visualization.TextElement):
    """
    Display a text count of how many steps have been taken
    """
    def __init__(self):
        pass

    def render(self, model):
        return "Steps: " + str(model.steps)
    

def households_draw(agent):
    """
    Portrayal Method for canvas
    """
    portrayal = dict()
    if not isinstance(agent, HouseholdAgent):
        return portrayal
    

    if agent.status == 'normal':
        #portrayal["dashArray"] = "1,3,2,2"
        portrayal["color"] = "Green"
    elif agent.status == 'displaced':
        #portrayal["dashArray"] = "1, 5"
        portrayal["color"] = "Black"
    elif agent.status == 'evacuated':
        #portrayal["dashArray"] = "1, 5"
        portrayal["color"] = "Red"

    agent_radius = 8.0
    if agent.income < POVERTY_LINE:
        agent_radius = 5.0

    portrayal["radius"] = agent_radius
    
    half_circle_length = agent_radius * np.pi
    house_damage = agent.house_damage * half_circle_length
    livelihood_damage = agent.livelihood_damage * half_circle_length
    house_not_damaged = half_circle_length - house_damage
    livelihood_not_damaged = half_circle_length - livelihood_damage

    portrayal["dashArray"] = f"{house_not_damaged}, {house_damage}, {livelihood_not_damaged}, {livelihood_damage}"


    #"Shape": Can be either "circle", "rect", "arrowHead"
    portrayal["description"] = agent.get_description()

    return portrayal


villages = [
    'Al-Gaili', 
    # 'Wawise Garb', 
    # 'Wad Ramli Camp', 
    # 'Eltomaniat', 
    # 'Al-Shuhada', 
    # 'Wawise Oum Ojaija', 
    # 'Wad Ramli'
]

events = get_events(initial_year=0, stride=MAX_YEARS)

all_settlements = gpd.read_file('IGAD/settlements_grid_wdst_sampled.gpkg').to_crs(epsg=4326)
bounding_boxes = gpd.read_file('IGAD/BoundingBox20022023/BoundingBox_20022023.shp').to_crs(epsg=4326)
# select only the bounding box of the village
all_population_data = load_population_data()

trusts = []
incomes = []
flood_prones = []
awarenesses = []
house_materials = []
obstacles_to_movement = []
fears = []
positions = []

for village in villages:
    bounding_box = bounding_boxes.query('village == @village').geometry
    settlements = all_settlements[all_settlements.geometry.within(bounding_box.unary_union)]
    # resample settlements to 1/10 of the original
    
    n_households = len(settlements)

    village_lons = settlements.geometry.centroid.x
    village_lats = settlements.geometry.centroid.y

    village_flood_prones = (village_lons - village_lons.min()) / (village_lons.max() - village_lons.min()) < 0.3
    village_flood_prones = village_flood_prones.values
    village_positions = list(zip(village_lons, village_lats))

    population_data = all_population_data\
            .query('village == @village')\
            .sample(n_households, replace=True)
        
    village_incomes = population_data['income'].apply(lambda x: (x + random())**1.3).values
    village_house_materials = population_data['walls_materials'].values
    village_fears = population_data['fear_of_flood'].values / 3
    village_obstacles_to_movement = (population_data[['vulnerabilities', 'properties']].sum(axis=1) > 4).values
    village_awarenesses = random(n_households)
    village_trusts = random(n_households)

    positions += village_positions
    trusts += village_trusts.tolist()
    incomes += village_incomes.tolist()
    flood_prones += village_flood_prones.tolist()
    awarenesses += village_awarenesses.tolist()
    house_materials += village_house_materials.tolist()
    obstacles_to_movement += village_obstacles_to_movement.tolist()
    fears += village_fears.tolist()


false_alarm_rate = 0.3
false_negative_rate = 0.0

model_params = dict(
    positions=positions,
    trusts=trusts,
    incomes=incomes,
    flood_prones=flood_prones,
    events=events,
    awarenesses=awarenesses,
    house_materials=house_materials,
    obstacles_to_movement=obstacles_to_movement,
    fears=fears,
    false_alarm_rate=mesa.visualization.Slider("False Alarm Rate", 0.3, 0, 1, 0.1),
    false_negative_rate=mesa.visualization.Slider("False Negative Rate", 0.1, 0, 1, 0.1),
)



model_text_element = IGADText()
map_element = mg.visualization.MapModule(
    households_draw,
    map_width=800,
    map_height=400,
)

chart_status = mesa.visualization.ChartModule([{
        "Label": "n_displaced",
        "Color": "Black"
    },{
        "Label": "n_evacuated",
        "Color": "Red"
    },{ 
        "Label": "n_normal",
        "Color": "Green"
    },{ 
        "Label": "n_flooded",
        "Color": "Blue"
    }],
    data_collector_name='datacollector'
)

chart_damage = mesa.visualization.ChartModule([
    {
        "Label": "mean_house_damage",
        "Color": "Red"
    },
    {
        "Label": "mean_livelihood_damage",
        "Color": "Green"
    },
    {
        "Label": "mean_perception",
        "Color": "Blue"
    },
    {
        "Label": "mean_trust",
        "Color": "Cyan"
    },
    {
        "Label": "mean_income",
        "Color": "Black"
    }

    ],
    data_collector_name='datacollector'
)


server = mesa.visualization.ModularServer(
    IGAD,
    [map_element, model_text_element, chart_status, chart_damage],
    "Basic agent-based IGAD model",
    model_params,
)
