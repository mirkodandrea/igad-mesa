from __future__ import annotations

import mesa
import numpy as np
import mesa_geo as mg
import rasterio as rio
from typing import List


class IGADCell(mg.Cell):
    water_level: float | None

    def __init__(
        self,
        pos: mesa.space.Coordinate | None = None,
        indices: mesa.space.Coordinate | None = None,
    ):
        super().__init__(pos, indices)
        self.water_level = None

    def step(self):
        pass


class IGADSpace(mg.GeoSpace):
    """
    Space for the IGAD model
    """

    def __init__(self, crs, reference, **kwargs):
        self._agents_positions = {}
        super().__init__(crs=crs, **kwargs)
        self.__init_water_level(reference)

    def __init_water_level(self, event_file: str):
        """
        Initialize the water level of the space using the first event as reference
        waterl_level is set to 0 for all cells
        """
        raster_layer = mg.RasterLayer.from_file(
            event_file, cell_cls=IGADCell, attr_name="water_level"
        )
        raster_layer.crs = 'epsg:4326'
        raster_layer.apply_raster(
            data=np.zeros(shape=(1, raster_layer.height, raster_layer.width)),
            attr_name="water_level",
        )
        super().add_layer(raster_layer)

    def get_water_level(self, agent: mg.GeoAgent):
        """check space for flood"""
        if agent in self._agents_positions:
            pos = self._agents_positions[agent]
        else:
            x, y = agent.geometry.xy
            i, j = (x[0],y[0]) * ~self.raster_layer.transform
            pos = round(i), round(j)
            self._agents_positions[agent] = pos
        
        cell = self.raster_layer[pos]

        return cell.water_level

    def reset_water_level(self):
        """
        Reset the water level of the space to 0
        """
        self.raster_layer.apply_raster(
            data=np.zeros(shape=(1, self.raster_layer.height, self.raster_layer.width)),
            attr_name="water_level",
        )

    def update_water_level(self, event_files: List[str]):
        """
        Update the water level of the space using the maximum water level for all events
        """

        flood_data = None
        for event_file in event_files:
            with rio.open(event_file) as f:
                if flood_data is None:
                    flood_data = f.read(1)
                else:
                    flood_data = np.maximum(flood_data, f.read(1))

        # add dimension to flood_data
        flood_data = np.expand_dims(flood_data, axis=0)

        self.raster_layer.apply_raster(
            data=flood_data, attr_name="water_level"
        )

    @property
    def raster_layer(self):
        return self.layers[0]

    def is_at_boundary(self, row_idx, col_idx):
        return (
            row_idx == 0
            or row_idx == self.raster_layer.height
            or col_idx == 0
            or col_idx == self.raster_layer.width
        )

    