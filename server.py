import mesa
import mesa_geo as mg

from model import IGAD
from agents import HouseholdAgent
from constants import (
    POVERTY_LINE, MAX_YEARS, ALPHA_INCOME
)
from utils import get_events, load_population_data

class IGADText(mesa.visualization.TextElement):
    """
    Display a text count of how many steps have been taken
    """
    def __init__(self):
        pass

    def render(self, model):
        return "Steps: " + str(model.steps)
    

from numpy.random import random, normal, poisson, pareto
import geopandas as gpd
import pandas as pd

events = get_events(initial_year=0, stride=MAX_YEARS)

settlements = gpd.read_file(
    'IGAD/settlements_with_price.gpkg').to_crs(epsg=4326)
population_data = load_population_data()

n_households = len(settlements)
lons = settlements.geometry.centroid.x
lats = settlements.geometry.centroid.y

flood_prones = (lons - lons.min()) / (lons.max() - lons.min()) < 0.3
positions = list(zip(lons, lats))

village = 'Al-Gaili'
population_data = population_data\
        .query('village == @village')\
        .sample(n_households, replace=True)
    
incomes = population_data['income'].apply(lambda x: (x + random())**1.3).values
house_materials = population_data['walls_materials'].values
fears = population_data['fear_of_flood'].values / 3
obstacles_to_movement = (population_data[['vulnerabilities', 'properties']].sum(axis=1) > 4).values
awarenesses = random(n_households)
trusts = random(n_households)

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



def households_draw(agent):
    """
    Portrayal Method for canvas
    """
    portrayal = dict()
    if isinstance(agent, HouseholdAgent):
        portrayal["radius"] = "5"

    # calculate color using rgb color space   
    r = int(255 * agent.house_damage)
    g = int(255 * (1 - agent.house_damage))
    b = 0

    portrayal["color"] = ('#%02x%02x%02x' % (r, g, b)).upper()

    if agent.status == 'normal':
        portrayal["dashArray"] = "1"
    elif agent.status == 'displaced':
        portrayal["dashArray"] = "1, 5"

    #"Shape": Can be either "circle", "rect", "arrowHead"
    portrayal["description"] = {
        'id': agent.unique_id,
        'damage': f"h: {int(100 * agent.house_damage)}% - l: {int(100 * agent.livelihood_damage)}%",
        'status': agent.status, 
        'income': f"{int(agent.income)}", 
        'awareness': f"{int(100 * agent.awareness)}%", 
        'fear': f"{int(100 * agent.fear)}%", 
        'perception': f"{int(100 * agent.perception)}%",
        'trust': f"{int(100 * agent.trust)}%", 
        'received_flood': agent.received_flood,
        'house_materials': agent.house_materials,
        'obstacles_to_movement': agent.obstacles_to_movement,
        'last_house_damage': f"{int(100 * agent.last_house_damage)}%",
        'last_livelihood_damage': f"{int(agent.last_livelihood_damage)}%",
    }

    return portrayal


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
