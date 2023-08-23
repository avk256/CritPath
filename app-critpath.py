 
# '''Streamlit app for strategy comparing'''

# import sys

# # base_path = '/home/avk256/Samawatt/trading-strategies/2021-03 Strategies exploration/apps/'
# # sys.path.append(base_path)

# import streamlit as st
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.offline import plot
# import numpy as np
# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt
# import os
# import base64
# import datetime
# import io

# import pygad
# import datetime
# import pdb

# from sklearn.metrics import mean_squared_error
# import random
# from datetimerange import DateTimeRange

# r3 = lambda x: f'{x:.3f}'

# CRIT_PATH = 0
# START_DATE = '01-01-21'
# END_DATE = '01-01-21'

# def cpa(data):
#     '''
#     Returns graph, nodes and attributes, critical path

#             Parameters:
#                     data (dataframe): dataframe

#             Returns:
#                     plt, atts, crt
#     '''


#     start = []
#     graph = []
#     atts = []
#     path = []
#     new = []
#     st = ""
   
#     last = data.iloc[-1, 0]
#     print(last)

#     last = chr(ord(last)+1)
#     print(last)
#     # -------------------------------------------
#     for j in range(len(data)):
#         for k in range(len(data.iloc[j, 1])):
#             if data.iloc[j, 1][k] != '-':
#                 new.append(data.iloc[j, 1][k])
#     # -------------------------------------------
#     for j in range(len(data)):
#         if not data.iloc[j, 0] in new:
#             st = st+data.iloc[j, 0]
#     # ------------------------------------------
#     if data.shape[1] == 3:
#         df = pd.DataFrame([[last, st, 0]], columns=["ac", "pr", "du"])
#     else:
#         df = pd.DataFrame([[last, st, 0, 0, 0]], columns=[
#                           "ac", "pr", "b", "m", "a"])
#     data = data.append(df)
#     for i in range(len(data)):
#         graph.append([])
#         atts.append({})
#     for j in range(len(data)):
#         atts[j]["Name"] = data.iloc[j, 0]
#         if data.shape[1] == 3:
#             atts[j]["DU"] = data.iloc[j, 2]
#         else:
#             atts[j]["DU"] = (data.iloc[j, 4] + 4 *
#                              data.iloc[j, 3] + data.iloc[j, 2]) / 6
#         if(data.iloc[j, 1] == "-"):
#             start.append(ord(data.iloc[j, 0])-65)
#             continue
#         for k in range(len(data.iloc[j, 1])):
#             graph[ord(data.iloc[j, 1][k]) -
#                   65].append(ord(data.iloc[j, 0])-65)

#     level = [None] * (len(graph))

#     def BFS(s, graph):
#         visited = [False] * (len(graph))
#         queue = []
#         for i in s:
#             queue.append(i)
#             level[i] = 0
#             visited[i] = True
#         while queue:
#             s = queue.pop(0)
#             path.append(s)
#             for i in graph[s]:
#                 if visited[i] == False:
#                     queue.append(i)
#                     level[i] = level[s] + 1
#                     visited[i] = True
#                 else:
#                     level[i] = max(level[s]+1, level[i])
#     BFS(start, graph)

#     levels = [None] * len(path)
#     for i in range(len(path)):
#         levels[i] = level[path[i]]
#     path = [x for y, x in sorted(zip(levels, path))]
#     print()
#     print("Path")
#     for i in path:
#         print(str(chr(i+65)), end=' ')
#     for s in path:
#         # print(str(chr(s+65)), " ", level[s])
#         # -------------Forward--------------------
#         if(data.iloc[s, 1] == "-"):
#             atts[s]["ES"] = 0
#         else:
#             ls = []
#             for k in range(len(data.iloc[s, 1])):
#                 ls.append(atts[ord(data.iloc[s, 1][k]) - 65]["EF"])
#             atts[s]["ES"] = max(ls)
#         atts[s]["EF"] = atts[s]["DU"] + atts[s]["ES"]
#         # ---------------------------------

#     for i in range(len(graph)):
#         if(graph[i] == []):
#             atts[i]["LF"] = atts[i]["EF"]
#             atts[i]["LS"] = atts[i]["ES"]
#     print()
#     print("------------------------")
#     # --------------------backward
#     path.reverse()
#     for i in path:
#         if(data.iloc[i, 1] != "-"):
#             for k in range(len(data.iloc[i, 1])):
#                 if "LF" in atts[ord(data.iloc[i, 1][k]) - 65].keys():
#                     atts[ord(data.iloc[i, 1][k]) - 65]["LF"] = min(atts[i]
#                                                                    ["LS"], atts[ord(data.iloc[i, 1][k]) - 65]["LF"])
#                 else:
#                     atts[ord(data.iloc[i, 1][k]) -
#                          65]["LF"] = atts[i]["LS"]
#                 atts[ord(data.iloc[i, 1][k]) - 65]["LS"] = atts[ord(data.iloc[i, 1]
#                                                                     [k]) - 65]["LF"] - atts[ord(data.iloc[i, 1][k]) - 65]["DU"]
#         atts[i]["SK"] = atts[i]["LF"] - atts[i]["EF"]
#     # ----------------------------------------
#     atts[-1]["Name"] = "End"
#     for j in range(len(graph)):
#         print(atts[j])
#     print()
#     # ------------------------------------------------
#     G2 = nx.DiGraph()

#     for i in range(len(graph)):
#         for j in graph[i]:
#             G2.add_edge(atts[i]["Name"], atts[j]["Name"])
#     temp = []
#     for i in range(len(atts)):
#         temp.append(atts[i]["Name"])
#     temp = dict(zip(temp, atts))
#     nx.set_node_attributes(G2, temp)
#     fig, ax = plt.subplots(figsize=(15, 15))
#     pos = nx.nx_agraph.graphviz_layout(G2, prog='dot')
#     # nx.draw(G2, pos=pos, ax=ax, with_labels=True, font_weight='bold')
#     nx.draw_networkx_edges(G2, pos, edge_color='olive',
#                            width=1, arrowstyle='simple', arrowsize=20, min_source_margin=25, min_target_margin=25)
#     crt = []
#     notcrt = []
#     for j, i in temp.items():
#         if(i["LF"] == i["EF"]):
#             crt.append(j)
#         else:
#             notcrt.append(j)
#     nx.draw_networkx_nodes(G2, pos, node_size=2000,
#                            node_color='seagreen', ax=ax, nodelist=crt)
#     nx.draw_networkx_nodes(G2, pos, node_size=1000,
#                            node_color='wheat', ax=ax, nodelist=notcrt)
#     nx.draw_networkx_labels(G2, pos, ax=ax, font_weight="bold",
#                             font_color="black", font_size=16)

#     def without(d, keys={"Name"}):
#         return {x: d[x] for x in d if x not in keys}

#     for node in G2.nodes:
#         xy = pos[node]
#         node_attr = G2.nodes[node]
#         d = G2.nodes[node]
#         d = without(d)
#         text = '\n'.join(f'{k}: {v}' for k,
#                          v in d.items())
#         ax.annotate(text, xy=xy, xytext=(50, 5), textcoords="offset points",
#                     bbox=dict(boxstyle="round", fc="lightgrey"),
#                     arrowprops=dict(arrowstyle="wedge"))
#     ax.axis('off')
#     # plt.savefig('/drive/My Drive/data_CPA/data/'+str(q)+".png")
#     return plt, atts, crt

# def crit_path_eval(table, solution):
#     '''
#     Returns critical path for initial data table

#             Parameters:
#                     filename (str): filename with initial table
#                                     ac,pr,du
#                                     A,-,18
#                                     B,-,21
#                                     C,-,24
#                                     D,B,18

#             Returns:
#                     crit_path_len (float): Binary string of the sum of a and b
#     '''
    
     
#     data = table
#     data['du'] = solution
#     data_cpa = data.copy()
#     data_cpa = data_cpa[['activity','prev','du']]
#     data_cpa.columns = ['ac','pr','du']
#     _, nodes, path  = cpa(data_cpa)
    
#     crit_path_len = nodes[len(nodes)-1]['LS']
#     return crit_path_len, path


# def price_eval(table):
#     '''
#     Price evaluation

#             Parameters:
#                     table (DataFrame): filename with initial table
#                                     ac,pr,du
#                                     A,-,18
#                                     B,-,21
#                                     C,-,24
#                                     D,B,18

#             Returns:
#                     crit_path_len (float): Binary string of the sum of a and b
#     '''
#     data = table
#     # data['price'] = data['du'] *
#     _, nodes, _  = cpa(data)
#     crit_path_len = nodes[len(nodes)-1]['LS']
#     return crit_path_len

# def get_max_duration(data, acts, act_col):
#     ends = []
#     for i in acts:
#         idx = data.index[data[act_col] == i]
#         ends.append(list(data['end'][idx])[0])
#     max_end = max(ends)
#     return max_end


# def price_day(df):
#     global END_DATE
#     min_day = min(df['start'])
#     max_day = max(df['end'])
#     END_DATE = max_day
#     period = pd.date_range(start=min_day, end=max_day)
#     cum_price = pd.DataFrame(0.0, index=np.arange(len(period)),
#                              columns=['price', 'start', 'end'])
    
#     for num, day in enumerate(period, 0):
#         # TODO check the data range    
#         for i in range(len(df)):
#                 if (day >= df['start'][i]) and (day < df['end'][i]):
#                     cum_price['price'][num] = cum_price['price'][num] + df['price'][i]
                
#         cum_price['start'][num] = day
#         cum_price['end'][num] = day        
    
#     return cum_price
    

# def create_gantt(df, budget, dur_col, start_time):
#     '''
#     Prepare dataframes for gantt plotting

#             Parameters:
#                     table (DataFrame): filename with initial table
#                                     ac,pr,du
#                                     A,-,18
#                                     B,-,21
#                                     C,-,24
#                                     D,B,18

#             Returns:
#                     crit_path_len (float): Binary string of the sum of a and b
#     '''

#     # print(df)
#     # print(budget)
#     # breakpoint()
    
#     start_time_obj = datetime.datetime.strptime(start_time, '%d-%m-%y')
#     days_list = [] 
#     df['end'] = 0
#     df['start'] = 0
#     for i in range(len(df)):
#         if df['prev'][i] == '-':
#             df['start'][i] = start_time_obj
            
#         else:
#             prev_act = df['prev'][i]
#             act_list = [act for act in prev_act]
#             if len(act_list) == 1:
#                 start_day = list(data['end'].loc[data['activity']==act_list[0]])
#                 df['start'][i] = start_day[0]
#             else:
#                 df['start'][i] = get_max_duration(df, prev_act, 'activity')  
#         df['end'][i] = df['start'][i] + datetime.timedelta(days=int(df['du'][i]))
#     df = df.sort_values(by=['start'])
    
#     #------- create prices DataFrame
    
#     prices = df[['price', 'start', 'end']]
    
#     # print(prices)
#     # breakpoint()

#     cum_prices = price_day(prices)
#     cum_prices['type'] = 'activity'
    
#     budget['start'] = 0
#     budget['end'] = 0
    
#     budget['start'][0] = start_time_obj
#     budget['end'][0] = budget['start'][0] + datetime.timedelta(days=int(budget['du'][0])) 

#     # print(budget)
#     # breakpoint()
    
#     for i in range(1, len(budget)):
#         budget['start'][i] = budget['end'][i-1]
#         budget['end'][i] = budget['start'][i] + datetime.timedelta(days=int(budget['du'][i]))

    
#     cum_budget = price_day(budget)
    
#     cum_budget['type'] = 'budget'
    
#     cum_prices = cum_prices.iloc[:-1 , :]
#     cum_budget = cum_budget.iloc[:-1 , :]
        
#     price_df = pd.concat([cum_prices, cum_budget[['price', 'start', 'end', 'type']]], axis=0).reset_index(drop=True) 

#     for i in range(len(price_df)):
#         price_df['end'][i] = price_df['end'][i] + datetime.timedelta(seconds=23*60*60+3599)  
        
                  
#     return df, price_df


# def new_prices(df, new_dur):
#     assert len(df) == len(new_dur)
#     new_pr = [0] * len(df)
#     # print(new_dur)
#     # breakpoint()

#     for i in range(len(df)):
#         t_min = new_dur[i] - new_dur[i] * 0.1
#         t_max = new_dur[i] + new_dur[i] * 0.1
#         h = (df['price_max'][i] - df['price_min'][i]) / (t_max - t_min)
#         delta_c = (t_max - new_dur[i]) * h
#         new_pr[i] = delta_c + df['price'][i]  
    
#     new_df = df.copy()
#     new_df['du'] = new_dur
#     new_df['price'] = new_pr
    
#     # print(new_df)
#     # breakpoint()

#     new_df = pd.DataFrame(new_df.reset_index(drop=True))
    
#     return new_df
   

# def metric(df, budget, solution):
#     new_df = new_prices(df, solution)
    
#     # print('metric before gantt')
#     # print(solution)
    
#     # breakpoint()

#     df, prices_df = create_gantt(new_df, budget, 'du', START_DATE)

#     # breakpoint()
    
#     sol = prices_df[prices_df['type']=='activity']['price']
#     bud = prices_df[prices_df['type']=='budget']['price']
    
#     if len(sol) == len(bud):
#         error = mean_squared_error(prices_df[prices_df['type']=='activity']['price'],
#                                prices_df[prices_df['type']=='budget']['price'])
#     if len(sol) != len(bud):
#         print('Different crit path length?')
#         error = 10**6
#     # print(error)
#     # breakpoint()
    
#     return error

    
# def set_init_population(init_sol, n_pop):
#     init_pop = []
#     for i in range(n_pop):
#         new_sol = [x + random.randrange(-5,5) for x in init_sol]
#         init_pop.append(new_sol)
        
#     # print(init_pop)
#     # breakpoint()

#     return init_pop
    
# def find_prev(lst, key):
#     out = []
#     for i, j in enumerate(lst):
#         if j == key:
#             out.append(i)
#     return out


# def listToString(s): 
    
#     # initialize an empty string
#     str1 = "" 
    
#     # return string  
#     return (str1.join(s))


# def data_convert(df):
#     import string
#     prev = [''] * len(df)
#     df['prev'] = prev 
#     jobs = list(string.ascii_uppercase)
#     df['act1'] = [x.split('-')[0] for x in df['activity']]
#     df['act2'] = [x.split('-')[1] for x in df['activity']]
#     df['act'] = df['activity']
#     df['activity'] = jobs[0:len(df)] 
#     for i in range(len(prev)):
#         if df['act1'][i] == '1':
#             prev[i] = ['-']
#         else:
#             ind = find_prev(list(df['act2']), df['act1'][i])
#             prev[i] = list(df['activity'][ind])
#         df['prev'][i] = listToString(prev[i]) 
    
#     # df['prev'] = 
#     return df


# def add_prices(df):
#     price_min = [0] * len(df)
#     price_max = [0] * len(df)
    
#     df['price_min'] = price_min 
#     df['price_max'] = price_max 

#     for i in range(len(df)):
#         df['price_min'][i] = df['price'][i] - df['price'][i] * 0.2
#         df['price_max'][i] = df['price'][i] + df['price'][i] * 0.2
#     return df


# def plot_prep(df):
#     '''
#     Prepare dataframes for job price plotting

#         Parameters:
#                 df (DataFrame): filename with initial table with cols:
#                                 activity,start,price,type

#         Returns:
#                 xy (DataFrame): dataframe with prices
#     '''

    
#     # x = prices[prices['type']=='activity']['start']
#     y1 = pd.DataFrame(df[df['type']=='activity']['price'])
#     y2 = pd.DataFrame(df[df['type']=='budget']['price'].reset_index(drop=True))
    
#     time_range = DateTimeRange(str(df['start'][0]), str(df['start'][len(df)-1]))
#     x = []
    
#     for value in time_range.range(datetime.timedelta(minutes=1)):
#         x.append(value)
    
#     x = pd.DataFrame(x)
#     y1 = y1.loc[y1.index.repeat(24*60)].reset_index(drop=True)
#     y2 = y2.loc[y2.index.repeat(24*60)].reset_index(drop=True)
    
#     xy = pd.concat([x,y1,y2], axis=1)
#     xy.columns = ['x', 'y1', 'y2']

#     return xy        
    

# def listToString2(s): 
    
#     # initialize an empty string
#     str1 = "" 
    
#     # traverse in the string  
#     for ele in s: 
#         str1 += ele + "-->"  
    
    
#     # return string  
#     return str1 + 'end' 

# def kostyl(df, col_name, status):
#     if status == 1:
#         df[col_name] = df[col_name].str.replace('-', '--')
#     else:
#         df[col_name] = df[col_name].str.replace('--', '-')
    
#     return df

# # ----------- Streamlit App Genetic algorithm ----------------------------------------------
# import locale

# #locale.setlocale(locale.LC_TIME, 'uk_UA.UTF8') 

# st.set_page_config(layout="wide")
# st.title('Генетична оптимізація розкладу робіт')

# st.sidebar.write("Оберіть два CSV файли: таблицю робіт та бюджетування, CSV")
# uploaded_file1 = st.sidebar.file_uploader("Завантажити таблицю робіт, CSV", type="csv", accept_multiple_files=False) 

# if uploaded_file1 is not None:
#     data = pd.read_csv(uploaded_file1)
#     data_old = data.copy() 
#     data = data_convert(data)
#     data = add_prices(data)
  
# uploaded_file2 = st.sidebar.file_uploader("Завантажити бюджет, CSV", type="csv", accept_multiple_files=False) 

# if uploaded_file2 is not None:
    
#     budget = pd.read_csv(uploaded_file2)
#     budget = budget.rename(columns={'du': 'Тривалість, дні', 'price': 'Вартість, тис. грн.'})
#     st.write("Таблиця бюджетування")
#     st.dataframe(budget)
#     budget = budget.rename(columns={'Тривалість, дні': 'du', 'Вартість, тис. грн.': 'price'})

# # # Create row, column, and value inputs
# # row = st.number_input('row', max_value=budget.shape[0])
# # col = st.number_input('column', max_value=budget.shape[1])
# # value = st.number_input('value')

# # # Change the entry at (row, col) to the given value
# # budget.values[row][col] = value

# # # And display the result!
# # st.dataframe(budget)

  
# num_generations = st.sidebar.number_input('Кількість генерацій', min_value=1, max_value=100, value=2, step=1)
# num_parents_mating = st.sidebar.number_input('Число батьківських рішень', min_value=1, max_value=10, value=2, step=1)
# # sol_per_pop = st.sidebar.number_input('sol_per_pop', min_value=1, max_value=5, value=2, step=1)
# crossover_type = st.sidebar.radio('Тип кроссовера:', ["одноточковий", "двоточковий", "рівномірний", "розсіяний"])
# crossover_probability = st.sidebar.slider('Ймовірність скрещування', min_value=0, max_value=100, value=60, step=10)
# mutation_type = st.sidebar.radio('Тип мутації:', ["випадкова", "обмін", "скремблінг", "інверсія"])
# mutation_probability = st.sidebar.slider('Ймовірність мутації гену', min_value=0, max_value=100, value=60, step=10)
# mutation_percent_genes = st.sidebar.slider('Частка генів у мутації', min_value=0, max_value=100, value=60, step=10)

# col1, col2 = st.columns(2)
# col1.header("Початковий розклад")
# col2.header("Оптимізований розклад")

# crossover_type_dict = {'одноточковий': 'single_point', 'двоточковий': 'two_points', 'рівномірний': 'uniform',  'розсіяний': 'scattered' }
# mutation_type_dict = {'випадкова': 'random', 'обмін': 'swap', 'скремблінг': 'scramble', 'інверсія': 'inversion' }

# submit = st.sidebar.button('Run')

# if submit:
    
    
#     if (uploaded_file1 is not None) and (uploaded_file2 is not None):
            
#         init_solution = list(data['du'])
#         init_population = set_init_population(init_solution, 20) 
#         best_solutions = []
        
#         df, prices = create_gantt(data, budget, 'du', START_DATE)
            
#         xaxes = list(pd.date_range(start=START_DATE, end=END_DATE))
        
#         print(list(range(len(xaxes))))
        
#         xaxes = [ datetime.datetime.strftime(x, '%d-%m-%y') for x in xaxes]
        
#         xaxes = [str(x) for x in xaxes]
        
#         # print(xaxes)
#         # breakpoint()
        
#         xy = plot_prep(prices)
        
#         # print(xy['x'])
#         # breakpoint()
        
            
#         fig_prices = go.Figure()
#         fig_prices.add_trace(go.Scatter(x=xy['x'], y=xy['y1'],
#             fill= 'tonexty',
#             mode='lines',
#             line_color='indigo', name="Вартість робіт"))
#         fig_prices.add_trace(go.Scatter(
#             x=xy['x'],
#             y=xy['y2'],
#             fill='tonexty', # fill area between trace0 and trace1
#             mode='lines', line_color='red', name="Бюджет"))
        
#         fig_prices.update_layout(
#             yaxis_title="Вартість, тис.грн."
#     )
#     #     fig_prices.update_layout(
#     #         xaxis = dict(
#     #         tickmode = 'array',
#     #         tickvals = list(range(len(xaxes))),
#     #         ticktext = xaxes
#     #     )
#     # )
#         fig_prices.update_xaxes(tickformat='%d-%m-%y')
    
            
#         fig_prices.write_html("fig_prices.html")
        
#         df = df.rename(columns={'act': 'Коди робіт'})
#         df = kostyl(df, 'Коди робіт', 1)
#         fig_df = px.timeline(df, x_start="start", x_end="end", y="Коди робіт", title="Діаграма Ганта робіт")
#         fig_df.update_yaxes(autorange="reversed") # otherwise tasks are listed from the bottom up
#         fig_df.update_xaxes(tickformat='%d-%m-%y')
#         df = kostyl(df, 'Коди робіт', 0)
#         df = df.rename(columns={'Коди робіт': 'act'})
        
#         fig_df.write_html("fig_df.html")
            
#         CRIT_PATH, path = crit_path_eval(df, init_solution)
        
        
#         # --------- Genetic algorithm
            
#         def fitness_func(solution, solution_idx):
            
#             new_crit_path, _ = crit_path_eval(data, solution)
#             error = metric(data, budget, solution)
#             error = error + (10**6) * (CRIT_PATH-new_crit_path)**2 
#             fitness = 1 / error 
#             return fitness
        
#         fitness_function = fitness_func
                
#         def on_fitness(ga_instance, population_fitness):
#             fitness_function
            
#         last_fitness = 0
#         def on_generation(ga_instance):
#             global last_fitness
#             print("Generation = {generation}".format(generation=ga_instance.generations_completed))
#             print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
#             print("Change     = {change}".format(change=ga_instance.best_solution()[1] - last_fitness))
#             last_fitness = ga_instance.best_solution()[1]
            
#         crit_path = []
        
#         for i in range(len(path)-1):
#             for j in range(len(data)):
#                 if data['activity'][j] == path[i]:
#                     crit_path.append(data['act'][j])
            
#         # --------------- Left columns
#         #   (1,2)(2,5)(5,6)(6,8)(8,9) critical path
        
#         col1.write("Таблиця робіт")
#         data_old = data_old.rename(columns={'activity':'Код роботи', 'du': 'Тривалість, дні', 'price': 'Вартість, тис. грн.'})
#         col1.dataframe(data_old)
#         data_old = data_old.rename(columns={'Код роботи':'activity', 'Тривалість, дні':'du', 'Вартість, тис. грн.':'price' })
        
#         col1.plotly_chart(fig_df, use_container_width=True)    
#         col1.plotly_chart(fig_prices, use_container_width=True)
#         print(path)
#         print(crit_path)
#         print(data)
#         # breakpoint()
#         col1.write("Початковий критичний шлях: ")
#         col1.write(listToString2(crit_path))
#         col1.write("Довжина критичного шляху: ")
#         col1.write(CRIT_PATH)
        
#         # col1.write("Parameters of the original solution : {solution}".format(solution=init_solution)) 
        
#         #########################
        
#         ga_instance = pygad.GA(num_generations=int(num_generations),
#                                 num_parents_mating=int(num_parents_mating),
#                                 fitness_func=fitness_function,
#                                 initial_population=init_population,
#                                 # sol_per_pop=int(sol_per_pop),
#                                 num_genes=len(init_solution),
#                                 # on_start=on_start,
#                                 # on_fitness=on_fitness,
#                                 # on_parents=on_parents,
#                                 # on_crossover=on_crossover,
#                                 # on_mutation=on_mutation,
#                                 on_generation=on_generation,
#                                 crossover_type=crossover_type_dict[crossover_type],
#                                 crossover_probability= crossover_probability/100 + 0.1,
#                                 mutation_type=mutation_type_dict[mutation_type],
#                                 mutation_probability= mutation_probability/100 + 0.1, # [0,1]
#                                 mutation_by_replacement=False, # True
#                                 mutation_percent_genes= mutation_percent_genes + 0.1, # [0,100] 
#                                 # on_stop=on_stop
#                                 save_solutions=True)
        
#         ga_instance.run()
        
#         # Returning the details of the best solution.
#         solution, solution_fitness, solution_idx = ga_instance.best_solution()
#         print("Parameters of the best solution : {solution}".format(solution=solution))
#         print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
#         print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
        
#         new_df = new_prices(data, solution)
        
#         opt_df, opt_prices_df = create_gantt(new_df, budget, 'du', START_DATE)
                
#         xy_opt = plot_prep(opt_prices_df)
        
#         fig_prices_opt = go.Figure()
#         fig_prices_opt.add_trace(go.Scatter(x=xy_opt['x'], y=xy_opt['y1'],
#             fill= 'tonexty',
#             mode='lines',
#             line_color='indigo', name="Вартість робіт"))
#         fig_prices_opt.add_trace(go.Scatter(
#             x=xy_opt['x'],
#             y=xy_opt['y2'],
#             fill='tonexty', # fill area between trace0 and trace1
#             mode='lines', line_color='red', name="Бюджет"))
        
#         fig_prices_opt.update_layout(
#         yaxis_title="Вартість, тис.грн."
#     )
#         fig_prices_opt.update_xaxes(tickformat='%d-%m-%y')
        
        
#         # plot(fig_prices_opt)
        
#         # fig_prices_opt = px.line(opt_prices_df, x="start", y="price", color="type", title='Optimal Prices vs Budget')
#         opt_df = opt_df.rename(columns={'act': 'Коди робіт'})
#         opt_df = kostyl(opt_df, 'Коди робіт', 1)
#         opt_fig_df = px.timeline(opt_df, x_start="start", x_end="end", y="Коди робіт", title="Діаграма Ганта робіт")
#         opt_fig_df.update_yaxes(autorange="reversed") # otherwise tasks are listed from the bottom up
#         opt_fig_df.update_xaxes(tickformat='%d-%m-%y')
#         opt_df = kostyl(opt_df, 'Коди робіт', 0)
#         opt_df = opt_df.rename(columns={'Коди робіт': 'act'})
        
        
#         fig_df.write_html("fig_df.html")
        
#         fig_prices_opt.write_html("fig_prices_opt.html")
#         # breakpoint()
            
#         opt_crit_path, opt_path = crit_path_eval(df, solution)
        
#         print(opt_crit_path)
#         solution_int = [int(x) for x in solution]
        
#         crit_path_opt = []
        
#         for i in range(len(opt_path)-1):
#             for j in range(len(opt_df)):
#                 if opt_df['activity'][j] == opt_path[i]:
#                     crit_path_opt.append(opt_df['act'][j])
    
        
#         # --------------- Right columns
        
#         opt_df['du'] = [int(x) for x in opt_df['du']]
#         col2.write("Таблиця робіт")
#         opt_df = opt_df.rename(columns={'act':'Код роботи', 'du': 'Тривалість, дні', 'price': 'Вартість, тис. грн.'})
#         col2.dataframe(opt_df[['Код роботи', 'Тривалість, дні', 'Вартість, тис. грн.']])
#         opt_df = opt_df.rename(columns={'act':'Код роботи', 'du': 'Тривалість, дні', 'price': 'Вартість, тис. грн.'})
        
#         col2.plotly_chart(opt_fig_df, use_container_width=True)    
#         col2.plotly_chart(fig_prices_opt, use_container_width=True)    
          
#         col2.write("Оптимізований критичний шлях: ")
        
#         col2.write(listToString2(crit_path_opt))
        
#         col2.write("Довжина критичного шляху: ")
#         col2.write(int(opt_crit_path))
        
#         col2.write("Зміна значень генів у кожному поколінні")
#         col2.plotly_chart(ga_instance.plot_genes(), use_container_width=True)
     
#         col1.write("Значення функції пристосованості для кожного покоління ")     
#         col1.plotly_chart(ga_instance.plot_fitness(), use_container_width=True)
        
#         col1.write("Кількість нових рішень, досліджених у кожному поколінні")
#         col1.plotly_chart(ga_instance.plot_new_solution_rate(), use_container_width=True)
        


 
'''Streamlit app for strategy comparing'''

import sys

# base_path = '/home/avk256/Samawatt/trading-strategies/2021-03 Strategies exploration/apps/'
# sys.path.append(base_path)

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import base64
import datetime
import io

import pygad

import pdb

from sklearn.metrics import mean_squared_error
import random
from datetimerange import DateTimeRange

import networkx as nx
import matplotlib.pyplot as plt
import ast

from criticalpath import Node




r3 = lambda x: f'{x:.3f}'

today = datetime.date.today()
year = today.strftime("%y")


CRIT_PATH = 0
START_DATE = '01-01-'+year
END_DATE = '01-01-'+year

# Функція для конвертації рядка в множину символів
def string_to_set(s):
    s1, s2 = s.split('-')[0], s.split('-')[1]
    return set([s1, s2])


def find_col(works, work_n):
    a = [ x for x  in works if x.split('-')[0]==str(work_n)]
    return a 

def lst_2_flat(llist):
    flat_list = [item for sublist in llist for item in sublist]
    return flat_list 

# Застосування функції до першого стовпця

def nodes_count(df):

    colname = 'at'
    # df = pd.read_csv(filename)
    df_set = list(df[colname].apply(string_to_set))
    df_merged_set = set().union(*df_set)
    df_int_set = {int(char) for char in df_merged_set}
    sorted_list = sorted(df_int_set)
    num_lists = sorted_list[len(sorted_list)-1] - 1

    # Створення списків за допомогою спискового компрехенша
    lists = [[] for _ in range(num_lists)]
    for i in range(len(lists)):
        lists[i].extend(find_col(list(df[colname]), i+1)) 
    
    # print(lists)
        
    flat_list = lst_2_flat(lists)

    return num_lists, flat_list, df
    
def list_2_tuple(flat_list):

    list_string = "[" + ", ".join(flat_list) + "]"
    # print(list_string)
    
    tmp = list_string.replace('[','[(')
    tmp = tmp.replace(']',')]')
    tmp = tmp.replace(',','),(')
    tmp = tmp.replace(' ','')
    tmp = tmp.replace('-',',')
    # print(tmp)

    try:
        path = ast.literal_eval(tmp)
        # print(path)
    except (SyntaxError, ValueError):
        print("Invalid input format")
        return 1

    return path

def path_nodes(max_n, graph_path, start, end):    
    # Створення графа
    G = nx.Graph()
    
    # Додавання вузлів і ребер
    G.add_nodes_from(list(range(1, max_n + 2)))
    G.add_edges_from(graph_path)
    
    # Візуалізація графа
    nx.draw(G, with_labels=True, node_color='lightblue', font_weight='bold')
    plt.show()
    
    # Побудова всіх можливих шляхів між вершинами
    start_node = start
    end_node = end
    all_paths = list(nx.all_simple_paths(G, source=start_node, target=end_node))

    def is_ascending(lst):
        return all(lst[i] < lst[i+1] for i in range(len(lst)-1))
        
    ascending_lists = [lst for lst in all_paths if is_ascending(lst)]
    
    return ascending_lists

def path_convert(path): 

    number_list = path
    
    duplicated_list = [number_list[0]] + [x for x in number_list[1:-1] for _ in range(2)] + [number_list[-1]]
    
    # print(duplicated_list)
    number_list = duplicated_list
    
    even_tuples = [(number_list[i], number_list[i+1]) for i in range(0, len(number_list), 2) if i+1 < len(number_list)]
    
    # print(even_tuples)
    
    return even_tuples

def path_convert2(paths, works_subs_dict): 

    new_paths = []
    for path in paths:

        list_string = ''.join(str(sublist) for sublist in path)
        list_string = list_string.replace(' ', '')
        
        # print(list_string)
        
        string = list_string
        for old_char, new_char in works_subs_dict.items():
            string = string.replace(old_char, new_char)
        
        string = '['+string+']'
        string = string.replace('][', '],[')
        
        
        string_with_lists = string
        
        # Видаляємо зовнішні дужки
        string_with_lists = string_with_lists.strip('[]')
        
        # Розділяємо рядок на підрядки списків
        sublists = string_with_lists.split('],[')
        
        # Перетворюємо підрядки у списки
        list_of_lists = [sublist.split(',') for sublist in sublists]
        
        # print(list_of_lists)
        
        tmp = [path_convert(lst) for lst  in  list_of_lists]
        # print(tmp)
        
        new_paths.append(tmp)
        
    return new_paths


def cpa(df):

    colname = 'at'
    print('before nodes_count \n', df)
    num_lists, nodes, df = nodes_count(df)
    print('after nodes_count \n', df)
    # pdb.set_trace()
    path = list_2_tuple(nodes)
    paths = []
    
    for i in range(1, num_lists+1):
        tmp = path_nodes(num_lists, path, i, num_lists+1) 
        path_node = [path_convert(lst) for lst  in  tmp]
        paths.append(path_node)
    
    
    
    alph = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 
           'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    var_name = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 
           'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    
    
    works = list(df[colname])
    df_subs_dict =dict(zip(works, alph) )
    
    works_u = [item.replace("-", ",") for item in works]
    works_list = ["(" + item + ")" for item in works_u]
    
    works_subs_dict = dict(zip(works_list, var_name) )
    
    rev_subs_dict =dict(zip(alph, works) )
    
    new_paths = path_convert2(paths, works_subs_dict)
    
    
    # df_subs_dict[df[colname][1]]
    
    pp = Node('project')
    
    for ind in range(len(df)):
        # print(df['activity'][ind])
        # print(var_name[ind])
        # print(df['du'][ind])
        # print((alph[ind], df['du'][ind]))
    
        if ind == 0: 
            globals()[var_name[ind]] = pp.add(Node(alph[ind], duration=df['du'][ind]))
        else:
            globals()[var_name[ind]] = pp.add(Node(alph[ind], duration=df['du'][ind], lag=0))
    
    # print(pp.name_to_node)
    
    
    for nodes_path in new_paths:
        for path in nodes_path:
            for tuple_node in path:
                # print(tuple_node)     
                if tuple_node[0] != tuple_node[1]:
                    pp.link(globals()[tuple_node[0]], globals()[tuple_node[1]]) 
    
    
    pp.update_all()
    
    cp = str(pp.get_critical_path())
    
    # print(cp)
    
    rev_cp = cp
    for old_char, new_char in rev_subs_dict.items():
        rev_cp = rev_cp.replace(old_char, new_char)
    
    print(rev_cp)
    
    print(pp.duration)

    return rev_cp, pp.duration


def crit_path_eval(table, solution):
    '''
    Returns critical path for initial data table

            Parameters:
                    filename (str): filename with initial table
                                    ac,pr,du
                                    A,-,18
                                    B,-,21
                                    C,-,24
                                    D,B,18

            Returns:
                    crit_path_len (float): Binary string of the sum of a and b
    '''
    
     
    data = table
    print(data)
    # pdb.set_trace()
    data['du'] = solution
    data_cpa = data.copy()
    data_cpa = data_cpa[['act', 'du']]
    data_cpa.columns = ['at','du']
    print(data_cpa)
    # pdb.set_trace()
    path, crit_path_len  = cpa(data_cpa)
    print(crit_path_len)
    # pdb.set_trace()
    return crit_path_len, path


def price_eval(table):
    '''
    Price evaluation

            Parameters:
                    table (DataFrame): filename with initial table
                                    ac,pr,du
                                    A,-,18
                                    B,-,21
                                    C,-,24
                                    D,B,18

            Returns:
                    crit_path_len (float): Binary string of the sum of a and b
    '''
    data = table
    # data['price'] = data['du'] *
    _, crit_path_len  = cpa(data)

    return crit_path_len

def get_max_duration(data, acts, act_col):
    ends = []
    for i in acts:
        idx = data.index[data[act_col] == i]
        ends.append(list(data['end'][idx])[0])
    max_end = max(ends)
    return max_end


def price_day(df):
    global END_DATE
    min_day = min(df['start'])
    max_day = max(df['end'])
    END_DATE = max_day
    period = pd.date_range(start=min_day, end=max_day)
    cum_price = pd.DataFrame(0.0, index=np.arange(len(period)),
                             columns=['price', 'start', 'end'])
    
    for num, day in enumerate(period, 0):
        # TODO check the data range    
        for i in range(len(df)):
                if (day >= df['start'][i]) and (day < df['end'][i]):
                    cum_price['price'][num] = cum_price['price'][num] + df['price'][i]
                
        cum_price['start'][num] = day
        cum_price['end'][num] = day        
    
    return cum_price
    

def create_gantt(df, budget, dur_col, start_time):
    '''
    Prepare dataframes for gantt plotting

            Parameters:
                    table (DataFrame): filename with initial table
                                    ac,pr,du
                                    A,-,18
                                    B,-,21
                                    C,-,24
                                    D,B,18

            Returns:
                    crit_path_len (float): Binary string of the sum of a and b
    '''

    # print(df)
    # print(budget)
    # breakpoint()
    # df = data.copy()
    print(data)
    print(df)
    # pdb.set_trace()
    start_time_obj = datetime.datetime.strptime(start_time, '%d-%m-%y')
    days_list = [] 
    df['end'] = 0
    df['start'] = 0
    for i in range(len(df)):
        if df['prev'][i] == '-':
            df['start'][i] = start_time_obj
            
        else:
            prev_act = df['prev'][i]
            act_list = [act for act in prev_act]
            if len(act_list) == 1:
                start_day = list(data['end'].loc[data['activity']==act_list[0]])
                df['start'][i] = start_day[0]
            else:
                df['start'][i] = get_max_duration(df, prev_act, 'activity')  
        df['end'][i] = df['start'][i] + datetime.timedelta(days=int(df['du'][i]))
    df = df.sort_values(by=['start'])
    
    #------- create prices DataFrame
    
    prices = df[['price', 'start', 'end']]
    
    # print(prices)
    # breakpoint()

    cum_prices = price_day(prices)
    cum_prices['type'] = 'activity'
    
    budget['start'] = 0
    budget['end'] = 0
    
    budget['start'][0] = start_time_obj
    budget['end'][0] = budget['start'][0] + datetime.timedelta(days=int(budget['du'][0])) 

    # print(budget)
    # breakpoint()
    
    for i in range(1, len(budget)):
        budget['start'][i] = budget['end'][i-1]
        budget['end'][i] = budget['start'][i] + datetime.timedelta(days=int(budget['du'][i]))

    
    cum_budget = price_day(budget)
    
    cum_budget['type'] = 'budget'
    
    cum_prices = cum_prices.iloc[:-1 , :]
    cum_budget = cum_budget.iloc[:-1 , :]
        
    price_df = pd.concat([cum_prices, cum_budget[['price', 'start', 'end', 'type']]], axis=0).reset_index(drop=True) 

    for i in range(len(price_df)):
        price_df['end'][i] = price_df['end'][i] + datetime.timedelta(seconds=23*60*60+3599)  
        
                  
    return data, df, price_df


def new_prices(df, new_dur):
    assert len(df) == len(new_dur)
    new_pr = [0] * len(df)
    # print(new_dur)
    # breakpoint()

    for i in range(len(df)):
        t_min = new_dur[i] - new_dur[i] * 0.1
        t_max = new_dur[i] + new_dur[i] * 0.1
        h = (df['price_max'][i] - df['price_min'][i]) / (t_max - t_min)
        delta_c = (t_max - new_dur[i]) * h
        new_pr[i] = delta_c + df['price'][i]  
    
    new_df = df.copy()
    new_df['du'] = new_dur
    new_df['price'] = new_pr
    
    # print(new_df)
    # breakpoint()

    new_df = pd.DataFrame(new_df.reset_index(drop=True))
    
    return new_df
   

def metric(df, budget, solution):
    new_df = new_prices(df, solution)
    
    # print('metric before gantt')
    # print(solution)
    
    # breakpoint()

    _,  df, prices_df = create_gantt(new_df, budget, 'du', START_DATE)

    # breakpoint()
    
    sol = prices_df[prices_df['type']=='activity']['price']
    bud = prices_df[prices_df['type']=='budget']['price']
    
    if len(sol) == len(bud):
        error = mean_squared_error(prices_df[prices_df['type']=='activity']['price'],
                               prices_df[prices_df['type']=='budget']['price'])
    if len(sol) != len(bud):
        print('Different crit path length?')
        error = 10**6
    # print(error)
    # breakpoint()
    
    return error

    
def set_init_population(init_sol, n_pop):
    init_pop = []
    for i in range(n_pop):
        new_sol = [x + random.randrange(-5,5) for x in init_sol]
        init_pop.append(new_sol)
        
    # print(init_pop)
    # breakpoint()

    return init_pop
    
def find_prev(lst, key):
    out = []
    for i, j in enumerate(lst):
        if j == key:
            out.append(i)
    return out


def listToString(s): 
    
    # initialize an empty string
    str1 = "" 
    
    # return string  
    return (str1.join(s))


def data_convert(df):
    import string
    prev = [''] * len(df)
    df['prev'] = prev 
    jobs = list(string.ascii_uppercase)
    df['act1'] = [x.split('-')[0] for x in df['activity']]
    df['act2'] = [x.split('-')[1] for x in df['activity']]
    df['act'] = df['activity']
    df['activity'] = jobs[0:len(df)] 
    for i in range(len(prev)):
        if df['act1'][i] == '1':
            prev[i] = ['-']
        else:
            ind = find_prev(list(df['act2']), df['act1'][i])
            prev[i] = list(df['activity'][ind])
        df['prev'][i] = listToString(prev[i]) 
    
    # df['prev'] = 
    return df


def add_prices(df):
    price_min = [0] * len(df)
    price_max = [0] * len(df)
    
    df['price_min'] = price_min 
    df['price_max'] = price_max 

    for i in range(len(df)):
        df['price_min'][i] = df['price'][i] - df['price'][i] * 0.2
        df['price_max'][i] = df['price'][i] + df['price'][i] * 0.2
    return df


def plot_prep(df):
    '''
    Prepare dataframes for job price plotting

        Parameters:
                df (DataFrame): filename with initial table with cols:
                                activity,start,price,type

        Returns:
                xy (DataFrame): dataframe with prices
    '''

    
    # x = prices[prices['type']=='activity']['start']
    y1 = pd.DataFrame(df[df['type']=='activity']['price'])
    y2 = pd.DataFrame(df[df['type']=='budget']['price'].reset_index(drop=True))
    
    time_range = DateTimeRange(str(df['start'][0]), str(df['start'][len(df)-1]))
    x = []
    
    for value in time_range.range(datetime.timedelta(minutes=1)):
        x.append(value)
    
    x = pd.DataFrame(x)
    y1 = y1.loc[y1.index.repeat(24*60)].reset_index(drop=True)
    y2 = y2.loc[y2.index.repeat(24*60)].reset_index(drop=True)
    
    xy = pd.concat([x,y1,y2], axis=1)
    xy.columns = ['x', 'y1', 'y2']

    return xy        
    

def listToString2(s): 

    res = ''.join(s)

    # return string  
    return res 


def kostyl(df, col_name, status):
        
    # Символ, який ви хочете додати

    
    # Додавання символу до кожного елементу вибраного стовпця
    if status == 1:
        df[col_name] = df[col_name].str.replace('-', '--')
    else:
        df[col_name] = df[col_name].str.replace('--', '-')
    
    return df

# ----------- Streamlit App Genetic algorithm ----------------------------------------------
import locale

#locale.setlocale(locale.LC_TIME, 'uk_UA.UTF8') 

st.set_page_config(layout="wide")
st.title('Генетична оптимізація розкладу робіт')

st.sidebar.write("Оберіть два CSV файли: таблицю робіт та бюджетування, CSV")
uploaded_file1 = st.sidebar.file_uploader("Завантажити таблицю робіт, CSV", type="csv", accept_multiple_files=False) 

if uploaded_file1 is not None:
    data = pd.read_csv(uploaded_file1)
    data_old = data.copy() 
    print('before data_convert \n', data)
    data = data_convert(data)
    data = add_prices(data)
    # print('after data_convert & add_price \n', data)
    
  
uploaded_file2 = st.sidebar.file_uploader("Завантажити бюджет, CSV", type="csv", accept_multiple_files=False) 

if uploaded_file2 is not None:
    
    budget = pd.read_csv(uploaded_file2)
    budget = budget.rename(columns={'du': 'Тривалість, дні', 'price': 'Вартість, тис. грн.'})
    st.write("Таблиця бюджетування")
    st.dataframe(budget)
    budget = budget.rename(columns={'Тривалість, дні': 'du', 'Вартість, тис. грн.': 'price'})

# # Create row, column, and value inputs
# row = st.number_input('row', max_value=budget.shape[0])
# col = st.number_input('column', max_value=budget.shape[1])
# value = st.number_input('value')

# # Change the entry at (row, col) to the given value
# budget.values[row][col] = value

# # And display the result!
# st.dataframe(budget)

  
num_generations = st.sidebar.number_input('Кількість генерацій', min_value=1, max_value=100, value=2, step=1)
num_parents_mating = st.sidebar.number_input('Число батьківських рішень', min_value=1, max_value=10, value=2, step=1)
# sol_per_pop = st.sidebar.number_input('sol_per_pop', min_value=1, max_value=5, value=2, step=1)
crossover_type = st.sidebar.radio('Тип кроссовера:', ["одноточковий", "двоточковий", "рівномірний", "розсіяний"])
crossover_probability = st.sidebar.slider('Ймовірність скрещування', min_value=0, max_value=100, value=60, step=10)
mutation_type = st.sidebar.radio('Тип мутації:', ["випадкова", "обмін", "скремблінг", "інверсія"])
mutation_probability = st.sidebar.slider('Ймовірність мутації гену', min_value=0, max_value=100, value=60, step=10)
mutation_percent_genes = st.sidebar.slider('Частка генів у мутації', min_value=0, max_value=100, value=60, step=10)

col1, col2 = st.columns(2)
col1.header("Початковий розклад")
col2.header("Оптимізований розклад")

crossover_type_dict = {'одноточковий': 'single_point', 'двоточковий': 'two_points', 'рівномірний': 'uniform',  'розсіяний': 'scattered' }
mutation_type_dict = {'випадкова': 'random', 'обмін': 'swap', 'скремблінг': 'scramble', 'інверсія': 'inversion' }

submit = st.sidebar.button('Run')

if submit:
    
    
    if (uploaded_file1 is not None) and (uploaded_file2 is not None):
            
        init_solution = list(data['du'])
        init_population = set_init_population(init_solution, 20) 
        best_solutions = []
        print(data)
        # pdb.set_trace()
        df, df_sort, prices = create_gantt(data, budget, 'du', START_DATE)
            
        xaxes = list(pd.date_range(start=START_DATE, end=END_DATE))
        
        print(list(range(len(xaxes))))
        
        xaxes = [ datetime.datetime.strftime(x, '%d-%m-%y') for x in xaxes]
        
        xaxes = [str(x) for x in xaxes]
        
        # print(xaxes)
        # breakpoint()
        
        xy = plot_prep(prices)
        
        # print(xy['x'])
        # breakpoint()
        
            
        fig_prices = go.Figure()
        fig_prices.add_trace(go.Scatter(x=xy['x'], y=xy['y1'],
            fill= 'tonexty',
            mode='lines',
            line_color='indigo', name="Вартість робіт"))
        fig_prices.add_trace(go.Scatter(
            x=xy['x'],
            y=xy['y2'],
            fill='tonexty', # fill area between trace0 and trace1
            mode='lines', line_color='red', name="Бюджет"))
        
        fig_prices.update_layout(
            yaxis_title="Вартість, тис.грн."
    )
    #     fig_prices.update_layout(
    #         xaxis = dict(
    #         tickmode = 'array',
    #         tickvals = list(range(len(xaxes))),
    #         ticktext = xaxes
    #     )
    # )
        fig_prices.update_xaxes(tickformat='%d-%m-%y')
    
            
        fig_prices.write_html("fig_prices.html")
        
        df = df.rename(columns={'act': 'Коди робіт'})
        df = df.astype({'Коди робіт':'string'})
        print('************************************* df ****************************')
        print(df)
        print(df.info())
        print('************************************* df ****************************')
        df = kostyl(df, 'Коди робіт', 1)
        # pdb.set_trace()
        fig_df = px.timeline(df, x_start="start", x_end="end", y="Коди робіт", title="Діаграма Ганта робіт")
        fig_df.update_yaxes(autorange="reversed") # otherwise tasks are listed from the bottom up
        fig_df.update_xaxes(tickformat='%d-%m-%y')
        df = kostyl(df, 'Коди робіт', 0)
        df = df.rename(columns={'Коди робіт': 'act'})
        
        fig_df.write_html("fig_df.html")
        
        CRIT_PATH, path = crit_path_eval(df, init_solution)
        print('---------------- crit path ------------------------')
        print(CRIT_PATH)
        print(path)
        # pdb.set_trace()        
        # --------- Genetic algorithm
            
        def fitness_func(solution, solution_idx):
            
            new_crit_path, _ = crit_path_eval(data, solution)
            error = metric(data, budget, solution)
            error = error + (10**6) * (CRIT_PATH-new_crit_path)**2 
            fitness = 1 / error 
            return fitness
        
        fitness_function = fitness_func
                
        def on_fitness(ga_instance, population_fitness):
            fitness_function
            
        last_fitness = 0
        def on_generation(ga_instance):
            global last_fitness
            print("Generation = {generation}".format(generation=ga_instance.generations_completed))
            print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
            print("Change     = {change}".format(change=ga_instance.best_solution()[1] - last_fitness))
            last_fitness = ga_instance.best_solution()[1]
            
        # crit_path = []
        
        # for i in range(len(path)-1):
        #     for j in range(len(data)):
        #         if data['activity'][j] == path[i]:
        #             crit_path.append(data['act'][j])
            
        # --------------- Left columns
        #   (1,2)(2,5)(5,6)(6,8)(8,9) critical path
        
        col1.write("Таблиця робіт")
        data_old = data_old.rename(columns={'activity':'Код роботи', 'du': 'Тривалість, дні', 'price': 'Вартість, тис. грн.'})
        col1.dataframe(data_old)
        data_old = data_old.rename(columns={'Код роботи':'activity', 'Тривалість, дні':'du', 'Вартість, тис. грн.':'price' })
        
        col1.plotly_chart(fig_df, use_container_width=True)    
        col1.plotly_chart(fig_prices, use_container_width=True)
        # breakpoint()
        col1.write("Початковий критичний шлях: ")
       
        # pdb.set_trace()
        col1.write(path)
        col1.write("Довжина критичного шляху: ")
        col1.write(CRIT_PATH)
        # pdb.set_trace()
        # col1.write("Parameters of the original solution : {solution}".format(solution=init_solution)) 
        
        #########################
        
        ga_instance = pygad.GA(num_generations=int(num_generations),
                                num_parents_mating=int(num_parents_mating),
                                fitness_func=fitness_function,
                                initial_population=init_population,
                                # sol_per_pop=int(sol_per_pop),
                                num_genes=len(init_solution),
                                # on_start=on_start,
                                # on_fitness=on_fitness,
                                # on_parents=on_parents,
                                # on_crossover=on_crossover,
                                # on_mutation=on_mutation,
                                on_generation=on_generation,
                                crossover_type=crossover_type_dict[crossover_type],
                                crossover_probability= crossover_probability/100 + 0.1,
                                mutation_type=mutation_type_dict[mutation_type],
                                mutation_probability= mutation_probability/100 + 0.1, # [0,1]
                                mutation_by_replacement=False, # True
                                mutation_percent_genes= mutation_percent_genes + 0.1, # [0,100] 
                                # on_stop=on_stop
                                save_solutions=True)
        
        ga_instance.run()
        
        # Returning the details of the best solution.
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
        
        new_df = new_prices(data, solution)
        
        _, opt_df, opt_prices_df = create_gantt(new_df, budget, 'du', START_DATE)
                
        xy_opt = plot_prep(opt_prices_df)
        
        fig_prices_opt = go.Figure()
        fig_prices_opt.add_trace(go.Scatter(x=xy_opt['x'], y=xy_opt['y1'],
            fill= 'tonexty',
            mode='lines',
            line_color='indigo', name="Вартість робіт"))
        fig_prices_opt.add_trace(go.Scatter(
            x=xy_opt['x'],
            y=xy_opt['y2'],
            fill='tonexty', # fill area between trace0 and trace1
            mode='lines', line_color='red', name="Бюджет"))
        
        fig_prices_opt.update_layout(
        yaxis_title="Вартість, тис.грн."
    )
        fig_prices_opt.update_xaxes(tickformat='%d-%m-%y')
        
        
        # plot(fig_prices_opt)
        
        # fig_prices_opt = px.line(opt_prices_df, x="start", y="price", color="type", title='Optimal Prices vs Budget')
        opt_df = opt_df.rename(columns={'act': 'Коди робіт'})
        opt_df = kostyl(opt_df, 'Коди робіт', 1)
        opt_fig_df = px.timeline(opt_df, x_start="start", x_end="end", y="Коди робіт", title="Діаграма Ганта робіт")
        opt_fig_df.update_yaxes(autorange="reversed") # otherwise tasks are listed from the bottom up
        opt_fig_df.update_xaxes(tickformat='%d-%m-%y')
        opt_df = kostyl(opt_df, 'Коди робіт', 0)
        opt_df = opt_df.rename(columns={'Коди робіт': 'act'})
        
        
        fig_df.write_html("fig_df.html")
        
        fig_prices_opt.write_html("fig_prices_opt.html")
        # breakpoint()
            
        opt_crit_path, opt_path = crit_path_eval(df, solution)
        
        print(opt_crit_path)
        solution_int = [int(x) for x in solution]
        
        # crit_path_opt = []
        
        # for i in range(len(opt_path)-1):
        #     for j in range(len(opt_df)):
        #         if opt_df['activity'][j] == opt_path[i]:
        #             crit_path_opt.append(opt_df['act'][j])
    
        
        # --------------- Right columns
        
        opt_df['du'] = [int(x) for x in opt_df['du']]
        col2.write("Таблиця робіт")
        opt_df = opt_df.rename(columns={'act':'Код роботи', 'du': 'Тривалість, дні', 'price': 'Вартість, тис. грн.'})
        col2.dataframe(opt_df[['Код роботи', 'Тривалість, дні', 'Вартість, тис. грн.']])
        opt_df = opt_df.rename(columns={'act':'Код роботи', 'du': 'Тривалість, дні', 'price': 'Вартість, тис. грн.'})
        
        col2.plotly_chart(opt_fig_df, use_container_width=True)    
        col2.plotly_chart(fig_prices_opt, use_container_width=True)    
          
        col2.write("Оптимізований критичний шлях: ")
        
        col2.write(opt_path)
        
        col2.write("Довжина критичного шляху: ")
        col2.write(int(opt_crit_path))
        
        col2.write("Зміна значень генів у кожному поколінні")
        col2.plotly_chart(ga_instance.plot_genes(), use_container_width=True)
     
        col1.write("Значення функції пристосованості для кожного покоління ")     
        col1.plotly_chart(ga_instance.plot_fitness(), use_container_width=True)
        
        col1.write("Кількість нових рішень, досліджених у кожному поколінні")
        col1.plotly_chart(ga_instance.plot_new_solution_rate(), use_container_width=True)
        










