from mesa.visualization.ModularVisualization import VisualizationElement, CHART_JS_FILE
import json

class GridLayoutModule(VisualizationElement):
    package_includes = [CHART_JS_FILE]
    local_includes = ["visualizers/GridLayoutModule.js"]

    def __init__(
        self,
        params=None,
    ):
        """
        Modify the layout of the visualization.
        Must be called before all the other modules.
        """
        params_str = json.dumps(params)

        new_element = f"new GridLayoutModule({params_str})"
        
        self.js_code = "elements.push(" + new_element + ");"

    def render(self, model):
        pass
        # current_values = []
        # data_collector = getattr(model, self.data_collector_name)

        # for s in self.series:
        #     name = s["Label"]
        #     try:
        #         val = data_collector.model_vars[name][-1]  # Latest value
        #     except (IndexError, KeyError):
        #         val = 0
        #     current_values.append(val)
        # return current_values