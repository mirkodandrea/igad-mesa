import pandas as pd
import rasterio as rio
MAPS_BASENAME = 'IGAD/Maps/SD_30mHazardMap'

def get_events(initial_year, stride=None):
    events = {}
    df_floods = pd.read_csv('IGAD/SD_EventCalendar.csv')
    df_floods = df_floods.query(
        'Year >= @initial_year and Year < @initial_year+@stride')

    for year, group in df_floods.groupby('Year'):
        for idx, row in group.iterrows():
            filename = f'{MAPS_BASENAME}_{row.EventID:0>4d}_cut.tif'
            with rio.open(filename) as f:
                flood_data = f.read(1)

            relative_year = year - initial_year
            if relative_year not in events:
                events[relative_year] = []

            events[relative_year].append(
                dict(
                    data=flood_data,
                    year=row.Year,
                    interarrival_time=row.InterarrivalTime,
                    rio_object=f,
                    filename=filename
                ))
    return events


def load_population_data() -> pd.DataFrame:
    df = pd.read_excel('IGAD/population_data.xlsx')
    df =  df[[
        'village',
        'income',
        'vulnerabilities',
        'properties',
        'walls_materials',
        'fear_of_flood',
        'household_size',
    ]]
    df.fillna(0, inplace=True)
    return df