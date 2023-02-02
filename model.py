import mesa
import mesa_geo as mg
from shapely.geometry import Point
import numpy as np
from agents import HouseholdAgent


class IGAD(mesa.Model):
    """Model class for a simplistic infection model."""


    def __init__(
        self, 
        positions=None,
        trusts=None,
        incomes=None,
        flood_prones=None,
        events=None,
        awarenesses=None,
        fears= None,
        false_alarm_rate=None,
        false_negative_rate=None,
    ):
        """
        Create a new InfectedModel
        :param pop_size:        Size of population
        :param init_infected:   Probability of a person agent to start as infected
        :param exposure_distance:   Proximity distance between agents to be exposed to each other
        :param infection_risk:      Probability of agent to become infected, if it has been exposed to another infected
        """
        self.schedule = mesa.time.BaseScheduler(self)
        self.space = mg.GeoSpace(crs='epsg:4326', warn_crs_conversion=False)
        self.steps = 0
        self.counts = None
        self.reset_counts()

        # IGAD MODEL PARAMETERS
        self.events = events
        self.false_alarm_rate = false_alarm_rate
        self.false_negative_rate = false_negative_rate

        
        
        self.running = True
        self.datacollector = mesa.DataCollector(
            {
                # "infected": get_infected_count,
                # "susceptible": get_susceptible_count,
                # "recovered": get_recovered_count,
                # "dead": get_dead_count,
            }
        )
        self.agents = []

        # Generate PersonAgent population
        ac_population = mg.AgentCreator(
            HouseholdAgent,
            model=self,
            crs=self.space.crs,
            agent_kwargs={},
        )
        # Generate random location, add agent to grid and scheduler
        n_agents = len(positions)
        for i in range(n_agents):
            x, y = positions[i]
            print(x,y)
            household = ac_population.create_agent(
                Point(x, y), 
                "H" + str(i)
            )
            household.trust = trusts[i]
            household.income = incomes[i]
            household.flood_prone = flood_prones[i]
            household.awareness = awarenesses[i]
            household.fear = fears[i]
            household.trust = trusts[i]

            self.space.add_agents(household)
            self.schedule.add(household)
            self.agents.append(household)

        self.datacollector.collect(self)

    def reset_counts(self):
        self.counts = {
            # "susceptible": 0,
            # "infected": 0,
            # "recovered": 0,
            # "dead": 0,
            # "safe": 0,
            # "hotspot": 0,
        }

    def __has_floods(self):
        return self.steps in self.events

    def maybe_emit_early_warning(self):
        """ 
        emit early warning.
        If there is a flood event in time t, emit early warning with probability 1 - false_negative_rate
        If there is no flood event in time t, emit early warning with probability false_alarm_rate        
        """
        emit = False
        if self.__has_floods():    
            emit = self.random.random() < self.false_negative_rate
        else:
            emit = self.random.random() < self.false_alarm_rate

        if emit:
            [agent.receive_early_warning() for agent in self.agents]
            [agent.check_neighbours() for agent in self.agents]
            pass

    def init_step(self):
        [agent.init_step() for agent in self.agents]

    def do_flood(self, events):
        """ 
        do flood
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



    def step(self):
        """Run one step of the model."""
        self.steps += 1
        self.reset_counts()
        

        self.init_step()
        self.maybe_emit_early_warning()

        if self.__has_floods():
            events = self.events[self.steps]
            self.do_flood(events)
        
        

        self.schedule.step()


        self.space._recreate_rtree()  # Recalculate spatial tree, because agents are moving

        self.datacollector.collect(self)

        # Run until no one is infected



# # Functions needed for datacollector
# def get_infected_count(model):
#     return model.counts["infected"]


# def get_susceptible_count(model):
#     return model.counts["susceptible"]


# def get_recovered_count(model):
#     return model.counts["recovered"]


# def get_dead_count(model):
#     return model.counts["dead"]
