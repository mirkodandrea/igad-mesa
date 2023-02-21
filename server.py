import mesa
import mesa_geo as mg

from model import IGAD
from agents import HouseholdAgent
from constants import (
    POVERTY_LINE, MAX_YEARS, ALPHA_INCOME
)
from utils import get_events

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

settlements = gpd.read_file(
    'IGAD/settlements_with_price.gpkg').to_crs(epsg=4326)
events = get_events(initial_year=0, stride=MAX_YEARS)
n_households = len(settlements)

lons = settlements.geometry.centroid.x
lats = settlements.geometry.centroid.y
flood_prones = (lons - lons.min()) / (lons.max() - lons.min()) < 0.3
positions = list(zip(lons, lats))
incomes = pareto(ALPHA_INCOME, n_households)

awarenesses = random(n_households)
fears = random(n_households)
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
    fears=fears,
    false_alarm_rate=mesa.visualization.Slider("False Alarm Rate", 0.3, 0, 1, 0.1),
    false_negative_rate=mesa.visualization.Slider("False Negative Rate", 0.1, 0, 1, 0.1),
)



# model_params = {
#     # "pop_size": mesa.visualization.Slider("Population size", 30, 10, 10000, 10),
#     # "init_infected": mesa.visualization.Slider(
#     #     "Fraction initial infection", 0.2, 0.00, 1.0, 0.05
#     # ),
#     # "exposure_distance": mesa.visualization.Slider(
#     #     "Exposure distance", 500, 100, 1000, 100
#     # ),
# }


def households_draw(agent):
    """
    Portrayal Method for canvas
    """
    portrayal = dict()
    if isinstance(agent, HouseholdAgent):
        portrayal["radius"] = "5"

    if agent.received_flood:
        portrayal["color"] = "Red"
    else:
        portrayal["color"] = "Green"

    if agent.status == 'normal':
        portrayal["dashArray"] = "1"

    # elif agent.status == 'evacuated':
    #     portrayal["color"] = "Yellow"

    elif agent.status == 'displaced':
        portrayal["dashArray"] = "1, 5"

    #"Shape": Can be either "circle", "rect", "arrowHead"
    portrayal["description"] = {
        'id': agent.unique_id,
        'damage': f"h: {agent.house_damage} - l: {agent.livelihood_damage}",
        'status': agent.status, 
        'income': int(agent.income), 
        'awareness': int(100 * agent.awareness), 
        'fear': int(100 * agent.fear), 
        'perception': int(100 * agent.perception),
        'trust': int(100 * agent.trust), 
        'received_flood': agent.received_flood
    }

    return portrayal


model_text_element = IGADText()
map_element = mg.visualization.MapModule(
    households_draw,
    map_width=400,
    map_height=400,
)

chart = mesa.visualization.ChartModule([{
    "Label": "n_displaced",
    "Color": "Black"}],
    data_collector_name='datacollector'
)

server = mesa.visualization.ModularServer(
    IGAD,
    [map_element, model_text_element, chart],
    "Basic agent-based IGAD model",
    model_params,
)
