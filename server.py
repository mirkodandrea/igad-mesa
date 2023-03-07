import mesa_geo as mg
import numpy as np

import mesa
from agents import (STATUS_DISPLACED, STATUS_EVACUATED, STATUS_NORMAL,
                    STATUS_TRAPPED, HouseholdAgent)
from constants import POVERTY_LINE
from model import IGAD, VILLAGES
from stacked_bar_chart import StackedBarChartModule


def households_draw(agent):
    """
    Portrayal Method for canvas
    """
    portrayal = dict()
    if not isinstance(agent, HouseholdAgent):
        return portrayal
    

    if agent.status == STATUS_NORMAL:
        portrayal["fillColor"] = "Green"
    elif agent.status == STATUS_DISPLACED:
        portrayal["fillColor"] = "Black"
    elif agent.status == STATUS_EVACUATED:
        portrayal["fillColor"] = "Red"
    elif agent.status == STATUS_TRAPPED:
        portrayal["fillColor"] = "Yellow"

    if agent.received_flood:
        portrayal["color"] = "Blue"
    else:
        portrayal["color"] = "Gray"

    agent_radius = 8.0
    if agent.income < POVERTY_LINE:
        agent_radius = 5.0

    if agent.prepared:
        portrayal['weight'] = 3.0
    else:
        portrayal['weight'] = 1.0

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




model_params = dict(
    _model_params=mesa.visualization.StaticText("Model Parameters"),
    
    false_alarm_rate=mesa.visualization.Slider("False Alarm Rate", 0.3, 0, 1, 0.1),
    false_negative_rate=mesa.visualization.Slider("False Negative Rate", 0.1, 0, 1, 0.1),
    trust=mesa.visualization.Slider("Authority Trust", 0.75, 0, 1, 0.05),
    government_help=mesa.visualization.Slider("Government Help", 0.0, 0, 1, 0.05),
    
    _events_params=mesa.visualization.StaticText("Events Parameters"),
    start_year=mesa.visualization.Slider("Start Year", 0, 0, 1000, 1),
    duration=mesa.visualization.Slider("Simulation Duration", 10, 5, 100, 1),
    
    _active_villages=mesa.visualization.StaticText("Active Villages"),
    ** {
        f'village_{n}': mesa.visualization.Checkbox(f"Village {village_name}", True) 
        for n, village_name in enumerate(VILLAGES)
    }
    
)

map_element = mg.visualization.MapModule(
    households_draw,
    map_width=300,
    map_height=900,
)

chart_status = StackedBarChartModule([
    {
        "Label": "n_normal",
        "Color": "Green"
    },
    {
        "Label": "n_displaced",
        "Color": "Black"
    },{
        "Label": "n_evacuated",
        "Color": "Red"
    }
    ,{ 
         "Label": "n_trapped",
         "Color": "Yellow"
    }],
    data_collector_name='datacollector',
    canvas_height=300, 
    canvas_width=1200
)

chart_affected = mesa.visualization.ChartModule([
    { 
        "Label": "n_flooded",
        "Color": "Blue"
    },
    {
        "Label": "affected_population",
        "Color": "Red"
    }
    ],
    data_collector_name='datacollector',
    canvas_height=300, 
    canvas_width=1200
)

chart_stats = mesa.visualization.ChartModule([
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
        "Label": "mean_awareness",
        "Color": "Magenta"
    },
    {
        "Label": "mean_fear",
        "Color": "Yellow"
    },
    ],
    data_collector_name='datacollector',
    canvas_height=300, 
    canvas_width=1200
)


chart_displacement = StackedBarChartModule([
    {
        "Color": "Red",
        "Label": "displaced_gt_5",
    },
    {
        "Color": "Orange",
        "Label": "displaced_lte_5",

    },
    {
        "Color": "Yellow",
        "Label": "displaced_lte_2",

    }
    ],
    data_collector_name='datacollector',
    canvas_height=300,  
    canvas_width=1200
)

from grid_layout import GridLayoutModule


gridParams = {
    "templateRows": 'repeat(5, 1fr)',
    "templateCols": '0.3fr repeat(4, 1fr)',
    "gridAreas": [
        '1 / 1 / 2 / 6',
        '2 / 1 / 6 / 3',
        '2 / 3 / 3 / 6',
        '3 / 3 / 4 / 6',
        '4 / 3 / 5 / 6',
        '5 / 3 / 6 / 6' 
    ]
}


server = mesa.visualization.ModularServer(
    IGAD,
    [GridLayoutModule(gridParams), map_element, chart_status, chart_affected, chart_stats, chart_displacement],
    "Agent-based IGAD model",
    model_params,
)
