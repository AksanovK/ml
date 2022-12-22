import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox
import random
from faker import Faker

if __name__ == '__main__':
    n = 100
    k = 10
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j] = np.random.randint(10, 50)
    g = nx.Graph()
    labels = np.arange(n)
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(labels[i], labels[j], weight=matrix[i][j])
    plt.figure(figsize=(30, 30))
    options = {
        'node_color': 'blue',
        'node_size': 12,
        'edge_color': 'grey',
        'width': 0.09,
        'label': True
    }
    nx.draw_kamada_kawai(g, **options)
    mst = nx.minimum_spanning_tree(g)
    plt.figure(figsize=(30, 30))
    options = {
        'node_color': 'blue',
        'node_size': 36,
        'edge_color': 'grey',
        'width': 0.5,
        'label': True
    }
    nx.draw_kamada_kawai(mst, **options)
    edges = sorted(mst.edges(data=True), key=lambda t: t[2].get('weight', 1))[:-k]
    edges = [(edge[0], edge[1], edge[2]['weight']) for edge in edges]
    clusters = nx.Graph()
    clusters.add_nodes_from(labels)
    clusters.add_weighted_edges_from(edges)
    plt.figure(figsize=(30, 30))
    options = {
        'node_color': 'blue',
        'node_size': 36,
        'edge_color': 'grey',
        'width': 0.5,
        'label': True
    }
    nx.draw(clusters, **options)
    place = {'city': 'Kazan',
             'country': 'Russia'}
    G = ox.graph_from_place(place, network_type='drive')
    Gp = ox.project_graph(G)
    points = ox.utils_geo.sample_points(ox.get_undirected(Gp), 257)
    nodes = list(G.nodes)
    fake = Faker()
    plt.savefig("edge_colormap.png")
    plt.show()
    for node_id in random.choices(nodes, k=500):
        print(
            f"""insert into driver_geo (on_the_road, lat, lon, timestamp, driver_id) values (rand() % 2, {G.nodes[node_id]['y']},{G.nodes[node_id]['x']},  
        '{fake.date_time_between(start_date='-1d', end_date='now').strftime("%Y-%m-%d %H:%M:%S")}', {random.choice(range(1, 50))});""")