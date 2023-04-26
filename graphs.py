#%%
import pandas as pd
import matplotlib.pyplot as plt
# %%


df_analysis = pd.read_csv('analysis.csv')
# %%
import numpy as np
def plot_spider_graph(data: pd.DataFrame, title: str):
    categories = data.index
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    
    ax = plt.subplot(111, polar=True)
    plt.title(title)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, color='grey', size=8)

    # Draw ylabels
    ax.set_rlabel_position(0)

    for column in data.columns:
        # cycle colors
        color = next(ax._get_lines.prop_cycler)['color']

        values = data[column].tolist()
        values += values[:1]
        # Plot data
        ax.plot(angles, values, linewidth=1, linestyle='solid', color=color, label=column)

        # Fill area
        #ax.fill(angles, values, color, alpha=0.5, label=column)

    max_value = data.max().max()
    plt.ylim(0, max_value + 5)
    # set ticks for y axis
    y_ticks = np.arange(0, max_value, max_value/5)
    y_ticks_str = [f'{y:.2f}' for y in y_ticks]
    plt.yticks(y_ticks, y_ticks_str, color="grey", size=7, rotation=45, ha='right')

    # add legend
    plt.legend(data.columns, loc='upper right', bbox_to_anchor=(0.1, 0.1))
    # rotate yticks 30 degrees



# %%
plt.style.use('ggplot')
programs = ['awareness_program', 'basic_income_program', 'house_improvement_program', 'do_early_warning']
metrics = ['n_trapped', 'n_evacuations', 'n_displacements', 'displaced_at_last_step', 'displacement_time<=2', 'displacement_time<=5', 'displacement_time>5']

for program in programs:
    plt.figure(figsize=(10, 5))
    if program == 'house_improvement_program':
        # consider 'house_repair_program'
        graph_values = pd.DataFrame()
        graph_values['No Improvement'] = df_analysis.query('house_improvement_program == False')[metrics].mean().T
        graph_values['Improve 10%'] = df_analysis.query('house_improvement_program == True & house_repair_program == 0.1')[metrics].mean().T
        graph_values['Improve 20%'] = df_analysis.query('house_improvement_program == True & house_repair_program == 0.2')[metrics].mean().T
        graph_values['Improve 30%'] = df_analysis.query('house_improvement_program == True & house_repair_program == 0.3')[metrics].mean().T
        graph_values['Improve 40%'] = df_analysis.query('house_improvement_program == True & house_repair_program == 0.4')[metrics].mean().T
        graph_values['Improve 50%'] = df_analysis.query('house_improvement_program == True & house_repair_program == 0.5')[metrics].mean().T
    else:
        graph_values = df_analysis.groupby(program)[metrics].mean().T

    graph_values.plot.bar()
    plt.grid(axis='x')
    # create a spider graph
    plt.polar()
    plt.legend(loc='upper right')

    plt.title(program)
    plt.savefig(f'graphs/{program}.png', dpi=300)

    plt.figure(figsize=(10, 5))    
    plot_spider_graph(graph_values, title=program)
    
    plt.savefig(f'graphs/{program}_spider.png', dpi=300)


# %%
