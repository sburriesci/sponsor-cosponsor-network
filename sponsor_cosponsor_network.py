# %pip install plotly --upgrade # must use plotly 6.0.0 or above

import pandas as pd
import numpy as np
import re
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly # must be version 6.0.0 or above
print(plotly.__version__)
import plotly.graph_objects as go
import csv
import os.path

# name the network chart you intend to build, based on subject collections of bills
# from https://www.congress.gov/browse/legislative-subject-terms
chart_title = "<br><b>Network of Income Tax Rate Bills, Sponsors and Cosponsors, 118th Congress</b>"

# set csv_fname for starting data
csv_fname = 'search_results_2025-01-28_0422pm.csv'

# create data directory path
data_dir = '/Data'

# create output directory path
out_dir = '/Output'

# read in data from CSV
# works with CSVs downloaded from https://www.congress.gov/browse/legislative-subject-terms
csvpath = os.path.join(data_dir, csv_fname)

df = pd.read_csv(csvpath, header=2)
df.drop(columns=['URL', 'Congress', 'Date of Introduction'], inplace=True)
print(df.columns)

################################### start functions #############################################
def remove_parens(text):
  """
  Remove parenthetical from text.

  Args:
    text: A string containing parentheses.

  Returns:
    A string with parentheses removed.
  """
  return re.sub(r'\([^)]*\)', '', text)

def group_bills_by_node(df, nodecol, colname):
  """
  Groups bill numbers by sponsor or cosponsor node ID.

  Args:
    df: A pandas DataFrame with columns nodecol (a column containing a node id) and colname (either the
    sponsor or cosponsor column).

  Returns:
    A dataframe where unique node IDs are associated with a list of bill numbers.
  """
  unique_nodes = df[nodecol].unique()
  node_bills = {}
  for node in unique_nodes:
    node_bills[node] = df[df[nodecol] == node][colname].tolist()
    node_bills[node] = ', '.join(sorted(list(set(node_bills[node]))))

  return pd.DataFrame(list(node_bills.items()), columns=[nodecol, 'bill_numbers'])
  

def draw_netx_nodes(node_list, node_shape):
  """
  Draws a group of Networkx nodes. Since different
  node types (sponsors, cosponsors, bills) have different shapes,
  we must draw each type of node separately.
  Args:
      node_list (list): A list of node IDs to draw.
      node_shape (str): The shape to use for node markers.
  Returns:
      None
  """
  nx.draw_networkx_nodes(G, pos, nodelist=node_list,
                         node_color=[node_colors[node] for node in node_list]
                         if isinstance(node_colors, dict)
                         else [node_colors[i] for i, node in enumerate(G.nodes)
                         if node in node_list], node_size=[node_sizes[node] for node in node_list]
                         if isinstance(node_sizes, dict)
                         else [node_sizes[i] for i, node in enumerate(G.nodes)
                         if node in node_list], node_shape=node_shape)
                         
############################## end functions #####################################################

# reshape the data
## melt the multiple cosponsor columns into one column,
## with one cosponsor per row
## (some bills have multiple cosponsors, others have none)
df_melted = df.melt(id_vars=['Legislation Number',
                             'Title',
                             'Sponsor'],
                    value_vars=[col for col in df.columns if col.startswith("Cosponsor")])
df_melted.columns = ['Legislation Number',
                     'title',
                     'sponsor',
                     'variable',
                     'cosponsor']

## drop rows that don't include a bill number
df_melted = df_melted.dropna(subset=['Legislation Number'])

## remove parentheticals from the sponsor column
df_melted['sponsor'] = df_melted['sponsor'].apply(remove_parens).str.strip()

## split sponsor and cosponsor columns into name and title_party_state
df_melted[['sponsor_name',
           'sponsor_title_party_state']] = df_melted['sponsor'].str.extract(r'(.+?)\s*(\[[^\]]+\])')
df_melted[['cosponsor_name',
           'cosponsor_title_party_state']] = df_melted['cosponsor'].str.extract(r'(.+?)\s*(\[[^\]]+\])')

## remove brackets from title_party_state
df_melted['sponsor_title_party_state'] = df_melted['sponsor_title_party_state'].astype(str).str.strip("[]").str.replace("'", "")
df_melted['cosponsor_title_party_state'] = df_melted['cosponsor_title_party_state'].astype(str).str.strip("[]").str.replace("'", "")

## split title_party_state to isolate party
df_melted[['sp_title',
           'sp_party',
           'sp_state',
           'sp_district']] = df_melted['sponsor_title_party_state'].str.split('-', expand=True)
df_melted[['cos_title',
           'cos_party',
           'cos_state',
           'cos_district']] = df_melted['cosponsor_title_party_state'].str.split('-', expand=True)
df_melted = df_melted.drop(columns=['sponsor_title_party_state',
                                    'cosponsor_title_party_state',
                                    'sp_district',
                                    'cos_district'])

## drop extra columns
df_melted = df_melted.drop(columns=['variable',
                                    'sp_title',
                                    'cos_title',
                                    'sp_state',
                                    'cos_state'
                                    ])

df_melted

# make a dataframe of unique bills & their sponsors
grouped = df_melted.groupby('Legislation Number')
bill_sponsor_df = grouped.first().reset_index()[['Legislation Number',
                                         'sponsor',
                                         'sp_party']]
bill_sponsor_df

## make a dataframe of bill nodes
bill_sponsor_df.columns = ['label', 'sponsor', 'party']
bill_sponsor_df['node_type'] = 'bill'
bill_sponsor_df['size'] = 128
## bill color will be based on sponsor's party
bill_sponsor_df['color'] = bill_sponsor_df['party'].apply(lambda x:
                                                          '#B22222' if
                                                          x == 'R' else
                                                          '#0047AB' if
                                                          x == 'D'
                                                          else '#FFAA33')
bill_sponsor_df['shape'] = 's' # all bill nodes will be square
bill_sponsor_df['sponsored'] = 'NA'
bill_sponsor_df['cosponsored'] = 'NA'
bill_sponsor_df['node_id'] = bill_sponsor_df.index + 1

# create a NetworkX graph
G = nx.Graph()

## add bill nodes to graph
for _, row in bill_sponsor_df.iterrows():
   G.add_node(row['node_id'], **row.drop('node_id').to_dict())

## print nodes with attributes
for node, attributes in G.nodes(data=True):
    print(f"Node: {node}, Attributes: {attributes}")

# create a dataframe of unique sponsors with node_ids; some legislators may sponsor more than one bill.
sponsor_df = df_melted[['sponsor', 'sp_party']].sort_values(by='sponsor').drop_duplicates().reset_index(drop=True)
sponsor_df.columns = ['label', 'party']
sponsor_df['sponsor'] = 'NA' # sponsors aren't sponsored
sponsor_df['node_type'] = 'sponsor'
sponsor_df['cosponsored'] = 'NA' # sponsors aren't cosponsored
sponsor_df['size'] = 128
sponsor_df['color'] = sponsor_df['party'].apply(lambda x:
                                                '#B22222' if
                                                x == 'R' else
                                                '#0047AB' if
                                                x == 'D'
                                                else '#FFAA33')
sponsor_df['shape'] = '^' # sponsor-only nodes will be triangles
sponsor_df['node_id'] = sponsor_df.index + len(G.nodes) + 1

## add sponsor nodes to graph
for _, row in sponsor_df.iterrows():
    G.add_node(row['node_id'], **row.drop('node_id').to_dict())

## filter for nodes with the attribute 'node_type' == 'sponsor'
sponsor_nodes = {node:
                 data for node,
                 data in G.nodes(data=True)
                 if data.get("node_type") == "sponsor"}

## print sponsor nodes with attributes
for node, attributes in sponsor_nodes.items():
    print(f"Node: {node}, Attributes: {attributes}")

## join sponsor_df to bill_df on sponsor name to get 'sponsored' bill(s)
sponsor_bill_df = sponsor_df.merge(bill_sponsor_df, left_on='label', right_on='sponsor')

## drop most bill_df columns
sponsor_bill_df = sponsor_bill_df[['node_id_x',
                                   'label_x',
                                   'party_x',
                                   'sponsor_x',
                                   'node_type_x',
                                   'cosponsored_x',
                                   'size_x',
                                   'color_x',
                                   'shape_x',
                                   'label_y',
                                   'node_id_y']]

## rename remaining columns
sponsor_bill_df.columns = ['sponsor_node_id',
                           'label',
                           'party',
                           'sponsor',
                           'node_type',
                           'cosponsored',
                           'size',
                           'color',
                           'shape',
                           'sponsored',
                           'bill_node_id']


## limit df to just sponsor_node_id, label, sponsored, bill_node_id
sponsor_bill_df = sponsor_bill_df[['sponsor_node_id',
                                   'label', 'sponsored',
                                   'bill_node_id']].sort_values(by='sponsor_node_id')

## create dataframe of sponsor nodes with lists of their sponsored bills
node_bills = group_bills_by_node(sponsor_bill_df, 'sponsor_node_id', 'sponsored')

## iterate through node_bills changing 'sponsored' attribute of sponsor nodes
for _, row in node_bills.iterrows():
  G.nodes[row['sponsor_node_id']]['sponsored'] = row['bill_numbers']

## verify modification
for node, attributes in sponsor_nodes.items():
    print(f"Node: {node}, Attributes: {attributes}")

# create edge list for bills and sponsors
## merge bill_nodes_df with sponsors_df
bill_sponsor_edges = sponsor_bill_df[['sponsor_node_id',
                                      'bill_node_id']].reset_index()
bill_sponsor_edges['style'] = '-.' # sponsorship edges will be dashdot
bill_sponsor_edges.rename(columns={'sponsor_node_id': 'source',
                                   'bill_node_id': 'target'}, inplace=True)

## add sponsor edges to graph
for _, row in bill_sponsor_edges.iterrows():
  G.add_edge(row['source'], row['target'], style=row['style'])

## print edge list
for target, source, attributes in G.edges(data=True):
    print(f"Edge: {(target, source)}, Attributes: {attributes}")

# make a dataframe of unique cosponsors; some legislators may cospsonsor more than one bill
## group by cosponsor

grouped = df_melted.groupby('cosponsor')
cosponsor_df = grouped[['cosponsor', 'cos_party']].apply(lambda x: x) # apply function to convert to dataframe
cosponsor_df = cosponsor_df.reset_index(drop=True)
cosponsor_df = cosponsor_df.drop_duplicates().reset_index(drop=True)
cosponsor_df['cosponsor'] = cosponsor_df['cosponsor'].str.strip()
cosponsor_df['size'] = 60
cosponsor_df['color'] = cosponsor_df['cos_party'].apply(lambda x:
                                                        '#B22222' if
                                                        x == 'R' else
                                                        '#0047AB' if
                                                        x == 'D'
                                                        else '#FFAA33')
cosponsor_df['shape'] = 'o' # cosponsor nodes will be circles
cosponsor_df.rename(columns={'cosponsor': 'label',
                             'cos_party': 'party'}, inplace=True)
                             
# check for overlap between cosponsors and sponsors
overlap_nodes_df = cosponsor_df.merge(sponsor_df,
                                            on='label',
                                            how='inner',
                                            suffixes=('_cosponsor', '_sponsor'))
overlap_nodes_df = overlap_nodes_df[['label', 'node_id']] # these nodes have already been added as sponsors

## join to df_melted to get cosponsored bill numbers
overlap_nodes_df = overlap_nodes_df.merge(df_melted,
                                          left_on='label',
                                          right_on='cosponsor')
overlap_nodes_df = overlap_nodes_df[['label',
                                     'node_id',
                                     'Legislation Number']].reset_index()
overlap_nodes_df.rename(columns={'node_id': 'cos_node_id',
                                 'Legislation Number': 'cosponsored'}, inplace=True)

## join to bill_sponsor_df to get bill node ids
overlap_nodes_df = overlap_nodes_df.merge(bill_sponsor_df,
                                          left_on='cosponsored',
                                          right_on='label')[['label_x',
                                                             'cos_node_id',
                                                             'cosponsored_x',
                                                             'node_id']]
overlap_nodes_df.rename(columns={'label_x': 'label',
                                 'cosponsored_x': 'cosponsored',
                                 'node_id': 'bill_node_id'}, inplace=True)

# create a dataframe of unique cosponsors-only (no sponsor/cosponsors) with node_ids
## make list of non-overlapping cosponsors
csonly_df = cosponsor_df[~cosponsor_df['label'].isin(overlap_nodes_df['label'])].reset_index()
## set node_type
csonly_df['node_type'] = 'cosponsor'
## create csonly_node_id
csonly_df['node_id'] = csonly_df.index + len(G.nodes) + 1
csonly_df.drop('index', axis=1, inplace=True)
csonly_df['sponsor'] = 'NA'
csonly_df['sponsored'] = 'NA'
csonly_df

## add cosponsor-only nodes to graph
for _, row in csonly_df.iterrows():
    G.add_node(row['node_id'], **row.drop('node_id').to_dict())

## filter for nodes with the attribute 'node_type' == 'sponsor'
cosponsor_nodes = {node:
                   data for node,
                   data in G.nodes(data=True)
                   if data.get("node_type") == "cosponsor"}

## print cosponsor nodes with attributes
for node, attributes in cosponsor_nodes.items():
    print(f"Node: {node}, Attributes: {attributes}")

## link bills to cosponsors
bill_cosponsor_df = df_melted.merge(csonly_df,
                                    left_on='cosponsor',
                                    right_on='label',
                                    suffixes=['_bill', '_cosponsor'])
bill_cosponsor_df.rename(columns={'Legislation Number': 'bill',
                                  'node_id': 'cs_node_id'}, inplace=True)
# add bill_node_id
bill_cosponsor_df = bill_cosponsor_df.merge(bill_sponsor_df,
                                            left_on='bill',
                                            right_on='label',
                                            suffixes=['_bill', '_cs'])
bill_cosponsor_df = bill_cosponsor_df[['bill',
                                       'node_id',
                                       'cosponsor',
                                       'cs_node_id', ]]
bill_cosponsor_df.rename(columns={'node_id': 'bill_node_id',
                                  'bill': 'cosponsored'}, inplace=True)

## make bill-cosponsor edgelist
bill_cosponsor_edges = bill_cosponsor_df[['bill_node_id',
                                          'cs_node_id']].reset_index()
bill_cosponsor_edges.rename(columns={'bill_node_id': 'source',
                                     'cs_node_id': 'target'}, inplace=True)
bill_cosponsor_edges['style'] = '-' # cosponsor edges will be solid
bill_cosponsor_edges.drop('index', axis=1, inplace=True)
bill_cosponsor_edges

# delete some unneeded dataframes
del grouped, df_melted

# create dataframe of cosponsor-only nodes with lists of their cosponsored bills
cs_node_bills = group_bills_by_node(bill_cosponsor_df, 'cs_node_id', 'cosponsored')
cs_node_bills

## iterate through node_bills changing 'cosponsored' attribute of cosponsor-only nodes
for _, row in cs_node_bills.iterrows():
  G.nodes[row['cs_node_id']]['cosponsored'] = row['bill_numbers']

## filter nodes for node_type 'cosponsor'
csonly_nodes = {node:
                data for node,
                data in G.nodes(data=True)
                if data.get("node_type") == "cosponsor"}

## verify modification
for node, attributes in csonly_nodes.items():
    print(f"Node: {node}, Attributes: {attributes}")

## add bill-cosponsor edges to graph
for _, row in bill_cosponsor_edges.iterrows():
  G.add_edge(row['source'], row['target'], style=row['style'])

## print edge list
for source, target, attributes in G.edges(data=True):
    print(f"Edge: {(source, target)}, Attributes: {attributes}")

# create dataframe of overlap nodes with lists of their cosponsored bills
node_bills = group_bills_by_node(overlap_nodes_df,
                                 'cos_node_id',
                                 'cosponsored')
print(node_bills)

## iterate through node_bills changing 'cosponsored' attribute of sponsor nodes
## shape, and node_type values
for _, row in node_bills.iterrows():
  G.nodes[row['cos_node_id']]['cosponsored'] = row['bill_numbers']
  G.nodes[row['cos_node_id']]['shape'] = 'H'
  G.nodes[row['cos_node_id']]['node_type'] = 'sponsor/cosponsor'

## filter nodes for node_type 'sponsor/cosponsor'
spcs_nodes = {node:
              data for node,
              data in G.nodes(data=True)
              if data.get("node_type") == "sponsor/cosponsor"}

## verify modification
for node, attributes in spcs_nodes.items():
    print(f"Node: {node}, Attributes: {attributes}")

## create edge dataframe for overlap nodes
overlap_edges_df = overlap_nodes_df[['bill_node_id', 'cos_node_id']]
overlap_edges_df = overlap_edges_df.rename(columns={'bill_node_id': 'source',
                                                    'cos_node_id': 'target'})
overlap_edges_df['style'] = '-' # these edges represent cosponsorship, solid
overlap_edges_df

## add cosponsorship edges for legislators who are both sponsors and cosponsors
for _, row in overlap_edges_df.iterrows():
  G.add_edge(row['source'], row['target'], style=row['style'])

## print edge list
for source, target, attributes in G.edges(data=True):
    print(f"Edge: {(source, target)}, Attribute: {attributes}")

# chart networkx graph, just to check basic structure (this plot will be ugly)
## we'll make a prettier plot with Plotly, below.
## Get node labels from the 'label' attribute
labels = nx.get_node_attributes(G, 'label')

## define node sizes based on node type, providing a default size if 'size' is missing
node_sizes = [G.nodes[node].get('size', 50) for node in G.nodes()]

## calculate the layout
pos = nx.spring_layout(G, k=.5)  # Adjust 'k' for spacing

## define node colors from node color attribute
node_colors = [G.nodes[node]['color']
               if 'color' in G.nodes[node]
               else 'gray' for node in G.nodes()]

## define node shapes
node_shapes = nx.get_node_attributes(G, 'shape') # dictionary of node:shape

## define edge styles
edge_styles = [G.edges[edge]['style']
               if 'style' in G.edges[edge]
               else '--' for edge in G.edges()]

## draw the graph
plt.figure(figsize=(15,10))

### create separate lists of nodes by shape
o_nodes = [node for node, shape in node_shapes.items() if shape == 'o']
s_nodes = [node for node, shape in node_shapes.items() if shape == 's']
t_nodes = [node for node, shape in node_shapes.items() if shape == '^']
h_nodes = [node for node, shape in node_shapes.items() if shape == 'H']

### draw each shape type separately
draw_netx_nodes(o_nodes, 'o')
draw_netx_nodes(s_nodes, 's')
draw_netx_nodes(t_nodes, '^')
draw_netx_nodes(h_nodes, 'H')

nx.draw_networkx_edges(G, pos, edge_color='#000000', style=edge_styles, width=0.5) # Draw edges separately
nx.draw_networkx_labels(G, pos, labels, font_size=9) # Draw labels separately

plt.show()

# export edges to csv
G_edges = nx.to_pandas_edgelist(G)
edge_csv_path = os.path.join(out_dir, 'G_edges.csv')
G_edges.to_csv(edge_csv_path, index=False)

# export nodes to csv
G_nodes = pd.DataFrame(list(G.nodes(data=True)), columns=['node', 'attributes'])
for i, row in G_nodes.iterrows():
    for key, value in row['attributes'].items():
        G_nodes.at[i, key] = value
G_nodes.drop(columns=['attributes'], inplace=True)
node_csv_path = os.path.join(out_dir, 'G_nodes.csv')
G_nodes.to_csv(node_csv_path, index=False)

# retrieve nodes and edges from CSV
# so you don't have to run all the previous code again to create a readable Plotly network diagram
node_path = os.path.join(out_dir, 'G_nodes.csv')
edge_path = os.path.join(out_dir, 'G_edges.csv')

nodes_df = pd.read_csv(node_path, keep_default_na=False)
print(nodes_df.head())

# create a new graph
Gfinal = nx.Graph()

## open the nodes CSV file
with open(node_path, 'r') as f:
    reader = csv.reader(f)
    next(reader) # Skip the header row if it exists

    ## add nodes to the graph
    for row in reader:
        node_id = row[0] # Assuming the first column in the CSV is the node ID
        Gfinal.add_node(node_id)
        nx.set_node_attributes(Gfinal, {node_id: {'label': row[1],
                                                  'sponsor': row[2],
                                                  'party': row[3],
                                                  'type': row[4],
                                                  'size': row[5],
                                                  'color': row[6],
                                                  'shape': row[7],
                                                  'sponsored': row[8],
                                                  'cosponsored': row[9]}})


## open the edges CSV file
with open(edge_path, 'r') as f:
    reader = csv.reader(f)
    next(reader) # Skip the header row if it exists

    for row in reader:
      Gfinal.add_edge(row[0], row[1], style=row[2])

## calculate node positions using spring_layout
pos = nx.spring_layout(Gfinal, k=0.15, iterations=20)

## add node positions to the graph
nx.set_node_attributes(Gfinal, pos, 'pos')

## Plotly code adapted from 'https://plotly.com/python/network-graphs/'
## create edge traces for each edge with its specific color
edge_traces = []
for edge in Gfinal.edges(data=True):
    x0, y0 = Gfinal.nodes[edge[0]]['pos']
    x1, y1 = Gfinal.nodes[edge[1]]['pos']

    # Map the 'style' attribute to a valid Plotly dash style
    dash_style = 'solid'  # Default to solid
    if edge[2]['style'] == '-':
        dash_style = 'solid'
    elif edge[2]['style'] == '-.':
        dash_style = 'dashdot'

    edge_trace = go.Scatter(
        x=[x0, x1, None],
        y=[y0, y1, None],
        name="Edges",
        line=dict(width=0.5, color='#000000', dash=dash_style),  # Use edge dash
        hoverinfo='none',
        showlegend=False,
        mode='lines'
    )
    edge_traces.append(edge_trace)  # Add trace to list

node_x = []
node_y = []
for node in Gfinal.nodes():
    x, y = Gfinal.nodes[node]['pos']
    node_x.append(x)
    node_y.append(y)

hover_text = [] ## create empty list to store hover text
included_attributes = ["label",
                       "sponsor",
                       "sponsored",
                       "cosponsored"]  ## list of node attributes to include in hover text

## append each node's hover text to list
for node in Gfinal.nodes():
    attributes = Gfinal.nodes[node]
    text = f""
    for attr_name, attr_value in attributes.items():
        if attr_name in included_attributes:
          if attr_value != "NA":
            text += f"{attr_name}: {attr_value}<br>"
    hover_text.append(text)

## create node trace
node_trace = go.Scatter(
    x=node_x, y=node_y,
    name='Nodes',
    mode='markers',
    hoverinfo='text',
    text=hover_text,
    showlegend=False,
    marker=dict(
        color=[c for c in nx.get_node_attributes(Gfinal, 'color').values()],
        size=[float(s)/4 for s in nx.get_node_attributes(Gfinal, 'size').values()],
        symbol=['square' if s == 's'
                else ('circle' if s == 'o' else 'triangle-down' if s == '^' else 'hexagon')
                for s in nx.get_node_attributes(Gfinal, 'shape').values()],
        line=dict(width=1, color='black'),
        showscale=False),
)

# create initial Plotly diagram
fig = go.Figure(data=edge_traces + [node_trace],
             layout=go.Layout(
                title=dict(
                    text=chart_title,
                    font=dict(
                        size=16
                    )
                ),
                hovermode='closest',
                margin=dict(b=20,l=10,r=0,t=55),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
fig.update_layout(width=1000, height=800, plot_bgcolor='#F1F1F1')

# create legend
## get a list of unique node colors for legend
unique_colors = set(nx.get_node_attributes(Gfinal, 'color').values())

## map the colors to the parties
color_map = {'#0047AB': 'Democrat', '#FFAA33': 'Independent', '#B22222': 'Republican'}

## create separate traces for each legend item
for color in unique_colors:
    color_legend_trace = go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            color=color,
            size=15,
            symbol='circle'  # Use circle for all legend items
        ),
        name=color_map.get(color, "Unknown"),  # Single party name
        showlegend=True
    )
    fig.add_trace(color_legend_trace)  # Add each legend trace separately

## add legend to chart
fig.update_layout(
    legend=dict(
        x=1.05,
        y=1.0,
        title="Legend",
        traceorder="normal",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        orientation="v",
        itemclick=False,
        font=dict(size=10)
    )
)

## get a list of unique node shapes for shape legend
unique_shapes = set(nx.get_node_attributes(Gfinal, 'shape').values())

## map the shapes to the node types
shape_map = {'s': 'Bill', 'o': 'Cosponsor', '^': 'Sponsor', 'H': 'Sponsor/Cosponsor'}

## map the shapes to the plotly symbol values
plotly_shape_map = {'s': 'square', 'o': 'circle', '^': 'triangle-down', 'H': 'hexagon'}

## create separate traces for each legend shape item
for shape in unique_shapes:
    # Get the Plotly symbol value, defaulting to 'circle' if not found in the map
    plotly_symbol = plotly_shape_map.get(shape, 'circle')

    shape_legend_trace = go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            color='white',  # Use black for all legend items
            size=15,
            symbol=plotly_symbol, # Use the mapped or default symbol
            line=dict(width=1, color='black')
        ),
        name=shape_map.get(shape, "Unknown"),
        showlegend=True
    )
    fig.add_trace(shape_legend_trace)  # Add each legend trace separately

## update legend to include shapes
fig.update_layout(
    legend=dict(
        x=1.05,
        y=.75,
        title="Legend",
        traceorder="normal",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        orientation="v",
        itemclick=False,
        font=dict(size=10)
    )
)

## add edge styles to legend
### map styles to edge types
dash_map = {'dashdot': 'sponsorship', 'solid': 'cosponsorship',}

### create separate traces for each legend item
for dash, label in dash_map.items():
    # get the linetype value
    plotly_symbol = dash_map.get(dash)

    dash_legend_trace = go.Scatter(
        x=[None],
        y=[None],
        mode='lines',
        line=dict( # changed from marker to line
            width=1,
            color='black',  # use black for all legend items
            dash=dash,
        ),
        name=label,
        showlegend=True
    )
    fig.add_trace(dash_legend_trace)  # add each legend trace separately

### update legend to include edge styles
fig.update_layout(
    legend=dict(
        x=1.05,
        y=.75,
        title="Legend",
        traceorder="normal",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        orientation="v",
        itemclick=False,
        font=dict(size=10)
    )
)

# set path for html output
html_path = os.path.join(out_dir, 'network.html')

# write interactive plot to html
fig.write_html(html_path)