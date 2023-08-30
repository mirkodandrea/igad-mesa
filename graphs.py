#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

#%%
# increase font size for plots
#plt.rcParams.update({'font.size': 22})


# def plot_spider_graph(data: pd.DataFrame, title: str):
#     plt.figure(figsize=figsize)
#     categories = data.index
#     N = len(categories)
#     angles = [n / float(N) * 2 * np.pi for n in range(N)]
#     angles += angles[:1]

#     # Initialise the spider plot
    
#     ax = plt.subplot(111, polar=True)
#     plt.title(title)


#     # Draw ylabels
#     ax.set_rlabel_position(0)

#     for column in data.columns:
#         # cycle colors
#         color = next(ax._get_lines.prop_cycler)['color']

#         values = data[column].tolist()
#         values += values[:1]
#         values = np.array(values)
#         # Plot data
#         ax.plot(angles, values, linewidth=1, linestyle='solid', color=color, label=column)

#         # Fill area
#         #ax.fill(angles, values, color, alpha=0.5, label=column)

#     max_value = data.max().max()
#     plt.ylim(0, max_value + 5)
#     # set ticks for y axis
#     tick_step = round(max_value/5)
#     # round to nearest 10    
#     tick_step = round(tick_step, -1)
#     if tick_step == 0:
#         tick_step = 1

#     y_ticks = np.arange(0, max_value, tick_step)
#     y_ticks_str = [f'{round(y)}' for y in y_ticks]
#     plt.yticks(y_ticks, y_ticks_str, color="grey", rotation=45, ha='right')
    

#     # Draw one axe per variable + add labels
#     plt.xticks(angles[:-1], categories, color='black', backgroundcolor='white', zorder=100 )

#     # add legend
#     plt.legend(data.columns, loc='upper right', bbox_to_anchor=(0.1, 0.1))
#     # rotate yticks 30 degrees



def plot_bar_graph(_graph_values: pd.DataFrame, figsize: tuple=None):
    graph_values = _graph_values.T.copy()

    # remove column if all values are zero
    graph_values = graph_values.loc[:, (graph_values != 0).any(axis=0)]
    n_columns = len(graph_values.columns)
    n_rows = min(len(graph_values.index)*2, 12)
    
    if figsize is None:
        figsize = (15, n_rows)
    
    # create a figure with a subplot for each column
    fig, ax = plt.subplots(nrows=1, ncols=n_columns, figsize=figsize, sharey=True)
    
    # change font size for all subplots
    for col_idx, column in enumerate(graph_values):
        graph_values.plot.barh(
            y=column, 
            ax=ax[col_idx], 
            legend=False, 
            color='#004070'
        )
        ax[col_idx].set_title(column, rotation=45)

        ax[col_idx].grid(True) #axis='y')
        # increment axes font size
        ax[col_idx].tick_params(axis='both', which='major', labelsize=20)


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
    'hrp_level+basic_income_program', 
    'ews_mode+basic_income_program',
    'ews_mode+awareness_program',
    'ews_mode+hrp_level',
]

metrics = ['n_trapped', 'n_evacuations', 'n_displacements', 'displaced_at_last_step', 'displacement_time<=2', 'displacement_time<=5', 'displacement_time>5']

# %%

def plot_graphs(df_analysis):
    all_graphs = pd.DataFrame()
    for scenario in scenarios:
        df_scenario = df_analysis.copy()
        programs = scenario.split('+')

        for column, value in baseline.items():
            if column not in programs:
                # select only baseline rows for the current column
                # since it is not part of the scenario
                df_scenario = df_scenario[df_scenario[column] == value]

        graph_values = pd.DataFrame()

        # get all permutations of possible program values
        programs_permutations = df_scenario[programs].drop_duplicates().values.tolist()
        baseline_column_name = None
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

            is_baseline = all(baseline[program] == value for program, value in zip(programs, program_values))
            if is_baseline:
                baseline_column_name = column_name
            
            graph_values[column_name] = df_scenario.query(query_str)[metrics].mean()

        graph_values['baseline'] = graph_values[baseline_column_name]
        #remove baseline column
        graph_values = graph_values.drop(columns=[baseline_column_name])

        graph_values = graph_values[['baseline'] + [c for c in graph_values if c not in ['baseline']]]
        plot_bar_graph(graph_values)
        plt.savefig(f'graphs/{scenario}.png', dpi=300, pad_inches=0.1, bbox_inches='tight')
        graph_values.to_csv(f'graphs/{scenario}.csv')
        

        all_graphs = pd.concat([all_graphs, graph_values], axis=1)
        # remove duplicated columns

    all_graphs = all_graphs.loc[:,~all_graphs.columns.duplicated()]
    # put baseline column last
    all_graphs = all_graphs[['baseline'] + [c for c in all_graphs if c not in ['baseline']]]

    plot_bar_graph(all_graphs, figsize=(15, 40))
    plt.savefig(f'graphs/all.png', dpi=300, pad_inches=0.1, bbox_inches='tight')
    graph_values.to_csv(f'graphs/all.csv')


    # %%
df_analysis = pd.read_csv('graphs/analysis.csv')
plot_graphs(df_analysis)