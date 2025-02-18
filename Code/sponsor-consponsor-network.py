%pip install plotly --upgrade # must use plotly 6.0.0 or above

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
data_dir = '/content/drive/MyDrive/Congress/Data'

# create output directory path
out_dir = '/content/drive/MyDrive/Congress/Output'

# read in data from CSV
# works with CSVs downloaded from https://www.congress.gov/browse/legislative-subject-terms
csvpath = os.path.join(data_dir, csv_fname)

df = pd.read_csv(csvpath, header=2)
df.drop(columns=['URL', 'Congress', 'Date of Introduction'], inplace=True)
print(df.columns)

# functions
def remove_parens(text):
  """
  Remove parenthetical from text.

  Args:
    text: A string containing parentheses.

  Returns:
    A string with parentheses removed.
  """
  return re.sub(r'\([^)]*\)', '', text)

def group_bills_by_node(df, colname='sponsored'):
  """
  Groups bill numbers by sponsor or cosponsor node ID.

  Args:
    df: A pandas DataFrame with columns 'node_id' and colname (default colname is 'sponsored').

  Returns:
    A dataframe where unique node IDs are associated with a list of bill numbers.
  """
  unique_nodes = df['node_id'].unique()
  node_bills = {}
  for node in unique_nodes:
    node_bills[node] = df[df['node_id'] == node][colname].tolist()
    node_bills[node] = ', '.join(sorted(list(set(node_bills[node]))))

  return pd.DataFrame(list(node_bills.items()), columns=['node_id', 'bill_numbers'])

# reshape the data
## melt the multiple cosponsor columns into one column
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
bill_df = grouped.first().reset_index()[['Legislation Number',
                                         'sponsor',
                                         'sp_party']]
print(bill_df)

# make a dataframe of bill nodes
bill_df.columns = ['label', 'sponsor', 'party']
bill_df['node_type'] = 'bill'
bill_df['size'] = 128
## bill color will be based on sponsor's party
bill_df['color'] = bill_df['party'].apply(lambda x: '#B22222' if x == 'R' else '#0047AB' if x == 'D' else '#FFAA33')
bill_df.drop(columns=['party'], inplace=True)
bill_df['shape'] = 's' # all bill nodes will be square
bill_df['sponsored'] = 'NA'
bill_df['cosponsored'] = 'NA'
bill_df['node_id'] = bill_df.index + 1

# create a NetworkX graph
G = nx.Graph()

# add bill nodes to graph
for _, row in bill_df.iterrows():
   G.add_node(row['node_id'], **row.drop('node_id').to_dict())

# print nodes with attributes
for node, attributes in G.nodes(data=True):
    print(f"Node: {node}, Attributes: {attributes}")

# create a dataframe of unique sponsors with node_ids; some legislators may sponsor more than one bill.
sponsor_df = df_melted[['sponsor', 'sp_party']].sort_values(by='sponsor').drop_duplicates().reset_index(drop=True)
sponsor_df.columns = ['label', 'party']
sponsor_df['sponsor'] = 'NA' # sponsors aren't sponsored
sponsor_df['node_type'] = 'sponsor'
sponsor_df['cosponsored'] = 'NA' # sponsors aren't cosponsored
sponsor_df['size'] = 128
sponsor_df['color'] = sponsor_df['party'].apply(lambda x: '#B22222' if x == 'R' else '#0047AB' if x == 'D' else '#FFAA33')
sponsor_df.drop(columns=['party'], inplace=True) # you don't need party once you have color
sponsor_df['shape'] = '^' # sponsor-only nodes will be triangles
sponsor_df['node_id'] = sponsor_df.index + len(G.nodes) + 1

# examine sponsor_df
sponsor_df

# add sponsor nodes to graph
for _, row in sponsor_df.iterrows():
    G.add_node(row['node_id'], **row.drop('node_id').to_dict())

# filter for nodes with the attribute 'node_type' == 'sponsor'
sponsor_nodes = {node: data for node, data in G.nodes(data=True) if data.get("node_type") == "sponsor"}

# print sponsor nodes with attributes
for node, attributes in sponsor_nodes.items():
    print(f"Node: {node}, Attributes: {attributes}")

# join sponsor_df to bill_df on sponsor name to get 'sponsored' bill(s)
sponsor_df = sponsor_df.merge(bill_df, left_on='label', right_on='sponsor', how='left')

# drop most bill_df columns
sponsor_df = sponsor_df[['label_x',
                         'sponsor_x',
                         'node_type_x',
                         'cosponsored_x',
                         'size_x',
                         'color_x',
                         'shape_x',
                         'node_id_x',
                         'label_y']]

# rename remaining columns
sponsor_df.columns = ['label',
                      'sponsor',
                      'node_type',
                      'cosponsored',
                      'size',
                      'color',
                      'shape',
                      'node_id',
                      'sponsored']

## drop party and reorder remaining sponsor_df columns
sponsor_df = sponsor_df[['label',
                         'sponsor',
                         'node_type',
                         'size',
                         'color',
                         'shape',
                         'sponsored',
                         'cosponsored',
                         'node_id']]

# examine revised sponsor_df
sponsor_df

# limit df to just label, sponsored
sponsored_bill_df = sponsor_df[['node_id', 'label', 'sponsored']].sort_values(by='node_id')

# create dataframe of sponsor nodes with lists of their sponsored bills
node_bills = group_bills_by_node(sponsored_bill_df, 'sponsored')
print(node_bills)

# iterate through node_bills changing 'sponsored' attribute of sponsor nodes
for _, row in node_bills.iterrows():
  G.nodes[row['node_id']]['sponsored'] = row['bill_numbers']

# verify modification
for node, attributes in sponsor_nodes.items():
    print(f"Node: {node}, Attributes: {attributes}")

# create edge list for bills and sponsors
## merge bill_nodes_df with sponsors_df
bill_sponsor_edges = bill_df.merge(sponsor_df[['label', 'node_id']],
                                   left_on='sponsor',
                                   right_on='label',
                                   how='left')
bill_sponsor_edges.drop(columns=['size', 'shape'], axis=1, inplace=True)
bill_sponsor_edges['color'] = 'cyan' # sponsorship edges will be cyan
bill_sponsor_edges.rename(columns={'node_id_x': 'target',
                                   'node_id_y': 'source'}, inplace=True)
bill_sponsor_edges = bill_sponsor_edges[['source', 'target', 'color']]

## add sponsor edges to graph
for _, row in bill_sponsor_edges.iterrows():
  G.add_edge(row['source'], row['target'], color=row['color'])

## print edge list
for target, source, attributes in G.edges(data=True):
    print(f"Edge: {(target, source)}, Attributes: {attributes}")

# make a dataframe of cosponsors
## group by cosponsor_name
grouped = df_melted.groupby('cosponsor')
cosponsor_df = grouped.first().reset_index()[['cosponsor',
                                              'cos_party',
                                              'Legislation Number']]
print('No. of Unique Cosponsors: ' + str(len(cosponsor_df['cosponsor'].unique())))

## limit dataframe to unique cosponsors
cosponsor_nodes_df = cosponsor_df['cosponsor'].str.strip().drop_duplicates().reset_index(drop=True)
cosponsor_nodes_df = pd.DataFrame(cosponsor_nodes_df)
cosponsor_nodes_df.columns = ['label']

## merge cosponsor_nodes_df with cosponsor_df to add party
cosponsor_nodes_df = cosponsor_nodes_df.merge(cosponsor_df,
                                              left_on='label',
                                              right_on='cosponsor').drop(['cosponsor'], axis=1)
cosponsor_nodes_df['size'] = 60
cosponsor_nodes_df['color'] = cosponsor_nodes_df['cos_party'].apply(lambda x: '#B22222' if x == 'R' else '#0047AB' if x == 'D' else '#FFAA33')
cosponsor_nodes_df['shape'] = 'o' # cosponsor nodes will be circles
cosponsor_nodes_df.rename(columns={'cos_party': 'party', 'Legislation Number': 'bill'}, inplace=True)
cosponsor_nodes_df.drop(columns=['party'], inplace=True)
cosponsor_nodes_df['node_id'] = cosponsor_nodes_df.index + len(G.nodes) + 1

# examine cosponsors
cosponsor_nodes_df

# delete some unneeded dataframes
del grouped, df_melted

# check for overlap between cosponsors and sponsors
overlap_nodes_df = cosponsor_nodes_df.merge(sponsor_df,
                                            left_on='label',
                                            right_on='label',
                                            how='inner',
                                            suffixes=('_cosponsor', '_sponsor'))
overlap_nodes_df = overlap_nodes_df[['label', 'bill', 'node_id_sponsor']]
overlap_nodes_df.columns = ['label', 'cosponsored', 'node_id']
print(overlap_nodes_df) # these nodes have already been added to the graph as sponsors

# make list of non-overlapping cosponsors
csonly_nodes_df = cosponsor_nodes_df[~cosponsor_nodes_df['label'].isin(overlap_nodes_df['label'])].reset_index()

# add node_id to csonly_nodes_df
csonly_nodes_df['node_id'] = csonly_nodes_df.index + len(G.nodes) + 1
csonly_nodes_df.rename({'bill': 'cosponsored'}, axis=1, inplace=True)
csonly_nodes_df['sponsor'] = 'NA'
csonly_nodes_df['sponsored'] = 'NA'
# reorder columns
csonly_nodes_df = csonly_nodes_df[['label',
                                   'sponsor',
                                   'size',
                                   'color',
                                   'shape',
                                   'sponsored',
                                   'cosponsored',
                                   'node_id']]
print(csonly_nodes_df)

# merge csonly_nodes_df with bill_nodes_df to add bill_node_id
csonly_nodes_df = csonly_nodes_df.merge(bill_df[['label', 'node_id']],
                                        left_on='cosponsored',
                                        right_on='label',
                                        how='left')
csonly_nodes_df.drop(columns=['label_y'], inplace=True)
csonly_nodes_df.columns = ['label',
                           'sponsor',
                           'size',
                           'color',
                           'shape',
                           'sponsored',
                           'cosponsored',
                           'target',
                           'source']

# reorder columns
csonly_nodes_df = csonly_nodes_df[['label',
                                   'sponsor',
                                   'size',
                                   'color',
                                   'shape',
                                   'sponsored',
                                   'cosponsored',
                                   'source',
                                   'target']]
csonly_nodes_df

# make bill-consponsor-only edge list
bill_cosponsor_edges = csonly_nodes_df[['source', 'target']].reset_index()
bill_cosponsor_edges['color'] = 'pink' # cosponsor edges will be pink
print(bill_cosponsor_edges)

# add cosponsor-only nodes to graph
csonly_nodes_df = csonly_nodes_df.rename(columns={'target': 'node_id'})
csonly_nodes_df['node_type'] = 'cosponsor'
print(csonly_nodes_df)

for _, row in csonly_nodes_df.iterrows():
    G.add_node(row['node_id'], **row.drop('node_id').to_dict())

# create dataframe of cosponsor-only nodes with lists of their cosponsored bills
node_bills = group_bills_by_node(csonly_nodes_df, 'cosponsored')
print(node_bills)

# iterate through node_bills changing 'cosponsored' attribute of cosponsor-only nodes
for _, row in node_bills.iterrows():
  G.nodes[row['node_id']]['cosponsored'] = row['bill_numbers']

# filter nodes for node_type 'cosponsor'
csonly_nodes = {node: data for node, data in G.nodes(data=True) if data.get("node_type") == "cosponsor"}

# verify modification
for node, attributes in csonly_nodes.items():
    print(f"Node: {node}, Attributes: {attributes}")

# print cosponsor only nodes
for node, attributes in csonly_nodes.items():
    print(f"Node: {node}, Attributes: {attributes}")

# add bill-cosponsor edges to graph
for _, row in bill_cosponsor_edges.iterrows():
  G.add_edge(row['source'], row['target'], color=row['color'])

# print edge list
for target, source, attributes in G.edges(data=True):
    print(f"Edge: {(target, source)}, Attributes: {attributes}")

# create dataframe of overlap nodes with lists of their cosponsored bills
node_bills = group_bills_by_node(overlap_nodes_df, 'cosponsored')
print(node_bills)

# iterate through node_bills changing 'cosponsored' attribute of sponsor nodes
# shape, and node_type values
for _, row in node_bills.iterrows():
  G.nodes[row['node_id']]['cosponsored'] = row['bill_numbers']
  G.nodes[row['node_id']]['shape'] = 'H'
  G.nodes[row['node_id']]['node_type'] = 'sponsor/cosponsor'

# filter nodes for node_type 'sponsor/cosponsor'
spcs_nodes = {node: data for node, data in G.nodes(data=True) if data.get("node_type") == "sponsor/cosponsor"}

# verify modification
for node, attributes in spcs_nodes.items():
    print(f"Node: {node}, Attributes: {attributes}")

# create edge dataframe for overlap nodes
overlap_nodes_df = overlap_nodes_df.merge(bill_df,
                                          left_on='cosponsored',
                                          right_on='label',
                                          how='left')
overlap_nodes_df = overlap_nodes_df[['node_id_x', 'node_id_y']]
overlap_nodes_df = overlap_nodes_df[['node_id_x', 'node_id_y']].rename(columns={'node_id_x': 'target',
                                                                                'node_id_y': 'source'})
overlap_nodes_df['color'] = 'pink' # these edges represent cosponsorship
overlap_nodes_df

# add cosponsorship edges for legislators who are both sponsors and cosponsors
for _, row in overlap_nodes_df.iterrows():
  G.add_edge(row['source'], row['target'], color=row['color'])

## print edge list
for target, source, attributes in G.edges(data=True):
    print(f"Edge: {(target, source)}, Attribute: {attributes}")

# chart networkx graph, just to check basic structure (this graph will be ugly)
## Get node labels from the 'label' attribute
labels = nx.get_node_attributes(G, 'label')

## Define node sizes based on node type, providing a default size if 'size' is missing
node_sizes = [G.nodes[node].get('size', 50) for node in G.nodes()]

## Calculate the layout
pos = nx.spring_layout(G, k=.5)  # Adjust 'k' for spacing

## Define node colors from node color attribute
node_colors = [G.nodes[node]['color'] if 'color' in G.nodes[node] else 'gray' for node in G.nodes()]

## Define node shapes
node_shapes = nx.get_node_attributes(G, 'shape') # dictionary of node:shape

## Define edge colors
edge_colors = [G.edges[edge]['color'] if 'color' in G.edges[edge] else 'gray' for edge in G.edges()]

## Draw the graph
plt.figure(figsize=(15,10))

### Separate nodes by shape
o_nodes = [node for node, shape in node_shapes.items() if shape == 'o']
s_nodes = [node for node, shape in node_shapes.items() if shape == 's']
t_nodes = [node for node, shape in node_shapes.items() if shape == '^']
h_nodes = [node for node, shape in node_shapes.items() if shape == 'H']

## Draw each shape type separately
nx.draw_networkx_nodes(G, pos, nodelist=o_nodes, node_color=[node_colors[node] for node in o_nodes] if isinstance(node_colors, dict) else [node_colors[i] for i, node in enumerate(G.nodes) if node in o_nodes], node_size=[node_sizes[node] for node in o_nodes] if isinstance(node_sizes, dict) else [node_sizes[i] for i, node in enumerate(G.nodes) if node in o_nodes], node_shape='o')
nx.draw_networkx_nodes(G, pos, nodelist=s_nodes, node_color=[node_colors[node] for node in s_nodes] if isinstance(node_colors, dict) else [node_colors[i] for i, node in enumerate(G.nodes) if node in s_nodes], node_size=[node_sizes[node] for node in s_nodes] if isinstance(node_sizes, dict) else [node_sizes[i] for i, node in enumerate(G.nodes) if node in s_nodes], node_shape='s')
nx.draw_networkx_nodes(G, pos, nodelist=t_nodes, node_color=[node_colors[node] for node in t_nodes] if isinstance(node_colors, dict) else [node_colors[i] for i, node in enumerate(G.nodes) if node in t_nodes], node_size=[node_sizes[node] for node in t_nodes] if isinstance(node_sizes, dict) else [node_sizes[i] for i, node in enumerate(G.nodes) if node in t_nodes], node_shape='^')
nx.draw_networkx_nodes(G, pos, nodelist=h_nodes, node_color=[node_colors[node] for node in h_nodes] if isinstance(node_colors, dict) else [node_colors[i] for i, node in enumerate(G.nodes) if node in h_nodes], node_size=[node_sizes[node] for node in h_nodes] if isinstance(node_sizes, dict) else [node_sizes[i] for i, node in enumerate(G.nodes) if node in h_nodes], node_shape='H')

nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=0.5) # Draw edges separately
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

# Create a new graph
Gfinal = nx.Graph()

# Open the nodes CSV file
with open(node_path, 'r') as f:
    reader = csv.reader(f)
    next(reader) # Skip the header row if it exists

    # Add nodes to the graph
    for row in reader:
        node_id = row[0] # Assuming the first column in the CSV is the node ID
        Gfinal.add_node(node_id)
        nx.set_node_attributes(Gfinal, {node_id: {'label': row[1], 'sponsor': row[2], 'type': row[3], 'size': row[4], 'color': row[5], 'shape': row[6], 'sponsored': row[7], 'cosponsored': row[8]}})

# Print the nodes in the graph
print(Gfinal.nodes(data=True))

# Open the eges CSV file
with open(edge_path, 'r') as f:
    reader = csv.reader(f)
    next(reader) # Skip the header row if it exists

    for row in reader:
      Gfinal.add_edge(row[0], row[1], color=row[2])

# Print the nodes in the graph
print(Gfinal.edges(data=True))

# Plotly code adapted from 'https://plotly.com/python/network-graphs/'
# calculate node positions using spring_layout
pos = nx.spring_layout(Gfinal, k=0.15, iterations=20)

# add node positions to the graph
nx.set_node_attributes(Gfinal, pos, 'pos')

# add edges as disconnected lines in a single trace and nodes as a scatter trace
edge_x = []
edge_y = []
for edge in Gfinal.edges():
    x0, y0 = Gfinal.nodes[edge[0]]['pos']
    x1, y1 = Gfinal.nodes[edge[1]]['pos']
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    name="Edges",
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    showlegend=False,
    mode='lines')

node_x = []
node_y = []
for node in Gfinal.nodes():
    x, y = Gfinal.nodes[node]['pos']
    node_x.append(x)
    node_y.append(y)

hover_text = [] # create empty list to store hover text
included_attributes = ["label", "sponsor", "sponsored", "cosponsored"]  # list of node attributes to include in hover text

for node in Gfinal.nodes():
    attributes = Gfinal.nodes[node]
    text = f""
    for attr_name, attr_value in attributes.items():
        if attr_name in included_attributes:
          if attr_value != "NA":
            text += f"{attr_name}: {attr_value}<br>"
    hover_text.append(text)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    name='Nodes',
    mode='markers',
    hoverinfo='text',
    text=hover_text,
    showlegend=False,
    marker=dict(
        color=[str(c) for c in nx.get_node_attributes(Gfinal, 'color').values()],
        size=[float(s)/4 for s in nx.get_node_attributes(Gfinal, 'size').values()],
        symbol=['square' if s == 's' else ('circle' if s == 'o' else 'triangle-down' if s == '^' else 'hexagon') for s in nx.get_node_attributes(Gfinal, 'shape').values()],
        line=dict(width=1, color='black'),
        showscale=False),
)

# Plotly code adapted from 'https://plotly.com/python/network-graphs/'
# create initial Plotly diagram
fig = go.Figure(data=[edge_trace, node_trace],
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

# get a list of unique node colors for legend
unique_colors = set(nx.get_node_attributes(Gfinal, 'color').values())

# map the colors to the parties
color_map = {'#0047AB': 'Democrat', '#FFAA33': 'Independent', '#B22222': 'Republican'}

# create separate traces for each legend item
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

# add legend to chart
fig.update_layout(
    legend=dict(
        x=1.05,
        y=1.0,
        title="Node Legend",
        traceorder="normal",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        orientation="v",
        itemclick=False,
        font=dict(size=10)
    )
)

# get a list of unique node shapes for shape legend
unique_shapes = set(nx.get_node_attributes(Gfinal, 'shape').values())

# map the shapes to the node types
shape_map = {'s': 'Bill', 'o': 'Cosponsor', '^': 'Sponsor', 'H': 'Sponsor/Cosponsor'}

# map the shapes to the plotly symbol values
plotly_shape_map = {'s': 'square', 'o': 'circle', '^': 'triangle-down', 'H': 'hexagon'}

# Create separate traces for each legend item
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

# update legend to incude shapes
fig.update_layout(
    legend=dict(
        x=1.05,
        y=.75,
        title="Node Legend",
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
