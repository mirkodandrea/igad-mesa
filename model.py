import mesa
import mesa_geo as mg
from shapely.geometry import Point
import numpy as np
from agents import HouseholdAgent, STATUS_EVACUATED, STATUS_NORMAL, STATUS_DISPLACED

EXPORT_TO_CSV = True
RAND_POSITION = False


class IGAD(mesa.Model):
    """Model class for the IGAD model."""
    def __init__(
        self, 
        positions=None,
        #trusts=None,
        incomes=None,
        flood_prones=None,
        events=None,
        awarenesses=None,
        fears= None,
        house_materials=None,
        obstacles_to_movement=None,
        false_alarm_rate=None,
        false_negative_rate=None,
        trust=None,
    ):
        """
        Create a new IGAD model.
        :param positions:   List of tuples with the x and y coordinates of each agent
        #:param trusts:      List of trust values for each agent
        :param incomes:     List of income values for each agent
        :param flood_prones:    List of flood prone values for each agent
        :param events:      List of events for each agent
        :param awarenesses: List of awareness values for each agent
        :param fears:       List of fear values for each agent
        :param house_materials: List of house material values for each agent
        :param obstacles_to_movement:   List of obstacles to movement values for each agent
        :param false_alarm_rate:    False alarm rate for the model
        :param false_negative_rate: False negative rate for the model
        """
        self.schedule = mesa.time.BaseScheduler(self)
        self.space = mg.GeoSpace(crs='epsg:4326', warn_crs_conversion=False)
        self.steps = 0
        self.counts = None

        # IGAD MODEL PARAMETERS
        self.events = events
        self.false_alarm_rate = false_alarm_rate
        self.false_negative_rate = false_negative_rate
        
        
        self.running = True
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "n_displaced": lambda this: len([a for a in this.agents if a.status == STATUS_DISPLACED]),
                "n_normal": lambda this: len([a for a in this.agents if a.status == STATUS_NORMAL]),
                "n_evacuated": lambda this: len([a for a in this.agents if a.status == STATUS_EVACUATED]),
                "n_flooded": lambda this: len([a for a in this.agents if a.received_flood]),
                "mean_house_damage": lambda this: np.mean([a.house_damage for a in this.agents]),
                "mean_livelihood_damage": lambda this: np.mean([a.livelihood_damage for a in this.agents]),
                "mean_trust": lambda this: np.mean([a.trust for a in this.agents]),
                "mean_perception": lambda this: np.mean([a.perception for a in this.agents]),
                "mean_income": lambda this: np.mean([a.income for a in this.agents]),
            },
            agent_reporters={
                "status": lambda agent: agent.status,
                "flooded": lambda agent: agent.received_flood,
                "alerted": lambda agent: agent.alerted,
                "house_damage": lambda agent: agent.house_damage,
                "livelihood_damage": lambda agent: agent.livelihood_damage,
                "trust": lambda agent: agent.trust,
                "perception": lambda agent: agent.perception,
                "income": lambda agent: agent.income,
            },
        )
        self.agents = []

        # Generate PersonAgent population
        ac_population = mg.AgentCreator(
            HouseholdAgent,
            model=self,
            crs=self.space.crs,
            agent_kwargs={},
        )

        # Create agents and assign them to the space, with slight randomization of the position
        n_agents = len(positions)

        for i in range(n_agents):
            x, y = positions[i]
            if RAND_POSITION:
                x = x + np.random.normal(0, 0.001)
                y = y + np.random.normal(0, 0.001)
            household = ac_population.create_agent(
                Point(x, y), 
                "H" + str(i)
            )
            # Assign attributes
            
            household.base_income = incomes[i]
            household.flood_prone = bool(flood_prones[i])
            household.awareness = awarenesses[i]
            household.fear = fears[i]
            #household.trust = trusts[i]
            household.trust = trust

            household.house_materials = house_materials[i]
            household.obstacles_to_movement = bool(obstacles_to_movement[i])

            self.space.add_agents(household)
            self.schedule.add(household)
            self.agents.append(household)

        self.datacollector.collect(self)

    def __has_floods(self):
        """
        Check if there is a flood event in current time step
        """
        return self.steps in self.events
    

    def init_step(self):
        """ execute init step for all agents
        """
        for household in self.agents:
            household.init_step()

    def maybe_emit_early_warning(self):
        """ 
        fuzzy emit early warning
        If there is a flood event in time t, emit early warning with probability 1 - false_negative_rate
        If there is no flood event in time t, emit early warning with probability false_alarm_rate        
        """
        emit = False
        if self.__has_floods():    
            emit = self.random.random() < self.false_negative_rate
        else:
            emit = self.random.random() < self.false_alarm_rate
        
        if not emit:
            return 

        print('Early warning at time step', self.steps)
        for agent in self.agents:
            agent.receive_early_warning()
                
        for agent in self.agents:                
            agent.check_neighbours_for_evacuation()
            
    def do_flood(self, events):
        """ 
        Apply flood to all agents
        """
        for household in self.agents:
            flood_value = 0
            for event in events:
                r = event['rio_object']
                flood_data = event['data']
                row, col = r.index(*household.geometry.xy)
                flood_value = np.nanmax(
                    flood_data[row, col]+
                    [flood_value]
                )
            household.receive_flood(flood_value)

    def fix_damages(self):
        """
        Fix damages for all agents
        """
        for agent in self.agents:
            agent.fix_damage()

        for agent in self.agents:
            agent.fix_neighbours_damage()

    def step(self):
        """Run one step of the model."""
        self.steps += 1      

        self.init_step()
        self.maybe_emit_early_warning()

        if self.__has_floods():
            events = self.events[self.steps]
            self.do_flood(events)

        self.fix_damages()

        # execute remaining steps for all agents
        self.schedule.step()
        self.datacollector.collect(self)
        
        if EXPORT_TO_CSV:            
            df = self.datacollector.get_agent_vars_dataframe()
            df.to_csv('output/data.csv')

