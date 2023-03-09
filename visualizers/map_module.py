from mesa_geo .visualization import MapModule

class MapModulePatched(MapModule):
    """
    Patched MapModule to override the local_includes
    """
    local_includes = [
        "visualizers/MapModule.js",
        "visualizers/leaflet.css",
        "visualizers/leaflet.js",
    ]
    local_dir = ""
    


