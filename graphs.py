#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

#%%
# increase font size for plots
#plt.rcParams.update({'font.size': 22})
figsize = (15, 12)

def plot_spider_graph(data: pd.DataFrame, title: str):
    plt.figure(figsize=figsize)
    categories = data.index
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    
    ax = plt.subplot(111, polar=True)
    plt.title(title)


    # Draw ylabels
    ax.set_rlabel_position(0)

    for column in data.columns:
        # cycle colors
        color = next(ax._get_lines.prop_cycler)['color']

        values = data[column].tolist()
        values += values[:1]
        values = np.array(values)
        # Plot data
        ax.plot(angles, values, linewidth=1, linestyle='solid', color=color, label=column)

        # Fill area
        #ax.fill(angles, values, color, alpha=0.5, label=column)

    max_value = data.max().max()
    plt.ylim(0, max_value + 5)
    # set ticks for y axis
    tick_step = round(max_value/5)
    # round to nearest 10    
    tick_step = round(tick_step, -1)
    if tick_step == 0:
        tick_step = 1

    y_ticks = np.arange(0, max_value, tick_step)
    y_ticks_str = [f'{round(y)}' for y in y_ticks]
    plt.yticks(y_ticks, y_ticks_str, color="grey", rotation=45, ha='right')
    

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, color='black', backgroundcolor='white', zorder=100 )

    # add legend
    plt.legend(data.columns, loc='upper right', bbox_to_anchor=(0.1, 0.1))
    # rotate yticks 30 degrees

def plot_bar_graph(graph_values: pd.DataFrame, title: str):
    plt.figure(figsize=figsize)    
    graph_values.plot.bar()
    plt.grid(axis='x')
    plt.title(title)
    plt.legend(loc='upper right')

# %%
baseline = {
    'awareness_program': False,
    'basic_income_program': False,
    'hrp_level': 'hrp_00',
    'ews_mode': 'no_ews',
}


scenarios = [
    'awareness_program', 
    'basic_income_program', 
    'hrp_level', 
    'ews_mode', 
    'basic_income_program+hrp_level', 
    'ews_mode+awareness_program',
    'ews_mode+hrp_level',
]
metrics = ['n_trapped', 'n_evacuations', 'n_displacements', 'displaced_at_last_step', 'displacement_time<=2', 'displacement_time<=5', 'displacement_time>5']

def plot_graphs(df_analysis):

    for scenario in scenarios:
        df_scenario = df_analysis.copy()
        programs = scenario.split('+')

        for column, value in baseline.items():
            if column in programs:
                continue
            df_scenario = df_scenario[df_scenario[column] == value]

        # normalize metrics columns
        # df_program[metrics] = 100 * df_program[metrics].div(df_program[metrics].max(axis=0), axis=1)

        graph_values = pd.DataFrame()

        # get all permutations of possible program values
        programs_permutations = df_scenario[programs].drop_duplicates().values.tolist()
        for program_values in programs_permutations:
            query_str = ' & '.join([f'{program} == {value.__repr__()}' for program, value in zip(programs, program_values)])

            column_names = []
            for (v, p) in zip(program_values, programs):
                if type(v) == str:
                    column_names.append(v)
                elif v:
                    column_names.append(f'{p}')
                else:
                    column_names.append(f'no_{p}')


            column_name = '+'.join(column_names)
            graph_values[column_name] = df_scenario.query(query_str)[metrics].mean()



        plot_bar_graph(graph_values, scenario)
        plt.savefig(f'graphs/{scenario}.png', dpi=300)

        plot_spider_graph(graph_values, title=scenario)    
        plt.savefig(f'graphs/{scenario}_spider.png', dpi=300)

if __name__ == '__main__':
    df_analysis = pd.read_csv('analysis.csv')
    plot_graphs(df_analysis)