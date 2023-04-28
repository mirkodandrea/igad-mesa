
import numpy as np
from typing import Tuple

import mesa
from agents import (STATUS_DISPLACED, STATUS_EVACUATED, STATUS_NORMAL,
                    STATUS_TRAPPED, HouseholdAgent)
from constants import POVERTY_LINE
from model import IGAD, VILLAGES, EWS_MODES, HOUSE_REPAIR_PROGRAMS_LEVELS

from visualizers.stacked_bar_chart import StackedBarChartModule
from visualizers.grid_layout import GridLayoutModule
from visualizers.map_module import MapModulePatched
from spaces import IGADCell
from utils import SCENARIOS, SimulationData


def portrayal(element: IGADCell|HouseholdAgent) -> dict|Tuple[float, float, float, float]:
    if isinstance(element, HouseholdAgent):
        return households_draw(element)
    elif isinstance(element, IGADCell):
        return cell_portrayal(element)
    else:
        raise ValueError("Unknown element type")

def cell_portrayal(cell: IGADCell) -> Tuple[float, float, float, float]:
    if cell.water_level == 0:
        return (0, 0, 0, 0)
    else:
        # return a blue color gradient based on the normalized water level
        # from the lowest water level colored as RGBA: (74, 141, 255, 1)
        # to the highest water level colored as RGBA: (0, 0, 255, 1)
        return (
            (1 - cell.water_level/1000) * 74,
            (1 - cell.water_level/1000) * 141,
            255,
            cell.water_level / 1000,
        )

def households_draw(agent):
    """
    Portrayal Method for Map
    """
    portrayal = dict()
    if not isinstance(agent, HouseholdAgent):
        return portrayal
    

    if agent.status == STATUS_NORMAL:
        portrayal["fillColor"] = "Green"
        portrayal["fillOpacity"] = "0.5"
    elif agent.status == STATUS_DISPLACED:
        portrayal["fillColor"] = "Black"
        portrayal["fillOpacity"] = "0.5"        
    elif agent.status == STATUS_EVACUATED:
        portrayal["fillColor"] = "Red"
        portrayal["fillOpacity"] = "0.5"
    elif agent.status == STATUS_TRAPPED:
        portrayal["fillColor"] = "Yellow"
        portrayal["fillOpacity"] = "0.5"

    
    portrayal["color"] = "Gray"
    # if agent.received_flood:
    #     portrayal["color"] = "Blue"
    # else:
    #     portrayal["color"] = "Gray"

    agent_radius = 8.0
    if agent.income < POVERTY_LINE:
        agent_radius = 5.0

    # if agent.prepared:
    #     portrayal['weight'] = 3.0
    # else:
    #     portrayal['weight'] = 1.0

    portrayal["radius"] = agent_radius
    half_circle_length = agent_radius * np.pi
    house_damage = agent.house_damage * half_circle_length
    livelihood_damage = agent.livelihood_damage * half_circle_length
    house_not_damaged = half_circle_length - house_damage
    livelihood_not_damaged = half_circle_length - livelihood_damage

    portrayal["dashArray"] = f"{house_not_damaged}, {house_damage}, {livelihood_not_damaged}, {livelihood_damage}"

    portrayal["description"] = agent.get_description()

    return portrayal


ews_modes = list(EWS_MODES.keys())
house_repair_programs_levels = list(HOUSE_REPAIR_PROGRAMS_LEVELS.keys())

model_params = dict(
    #save_to_csv=mesa.visualization.Checkbox("Save to CSV", True),
    
    _separator_1=mesa.visualization.StaticText("_______________________________"),
    _model_params=mesa.visualization.StaticText("Model Parameters"),
    
    ews_mode=mesa.visualization.Choice("EWS Mode", ews_modes[0], ews_modes),
    hrp_level=mesa.visualization.Choice("House Repair Program Level", house_repair_programs_levels[0], house_repair_programs_levels),

    basic_income_program=mesa.visualization.Checkbox("Basic Income Program", False),
    awareness_program=mesa.visualization.Checkbox("Awareness Program", False),
    
    _separator_2=mesa.visualization.StaticText("_______________________________"),
    _events_params=mesa.visualization.StaticText("Events Parameters"),
    scenario=mesa.visualization.Choice("Scenario", SCENARIOS[0], SCENARIOS),
    _active_villages=mesa.visualization.StaticText("Active Villages"),
    ** {
        f'village_{n}': mesa.visualization.Checkbox(f"{village_name}", True) 
        for n, village_name in enumerate(VILLAGES)
    }
    
)

map_element = MapModulePatched(
    portrayal,
    map_width=350,
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

IGAD.simulation_data = SimulationData()

server = mesa.visualization.ModularServer(
    IGAD,
    [GridLayoutModule(gridParams), map_element, chart_status, chart_affected, chart_stats, chart_displacement],
    "Agent-based IGAD model",
    model_params,
)
