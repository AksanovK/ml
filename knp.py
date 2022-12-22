import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def print_matrix(matrix):
    for i in range(len(matrix)):
        print(matrix[i])


def draw_graph(adj, N, clustered_peaks=None):
    G = nx.Graph()
    for i in range(len(adj)):
        G.add_node(i)
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i][j] != 0:
                G.add_edge(i, j)
    edge_labels = {}
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i][j] != 0:
                edge_labels[(i, j)] = adj[i][j]
    pos = nx.shell_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    if clustered_peaks is not None:
        for i in range(len(clustered_peaks)):
            current_vertex_cluster = clustered_peaks[i]
            x, y = pos[i]
            plt.text(x, y + 0.1, s=str(current_vertex_cluster), bbox=dict(facecolor='red', alpha=0.5),
                     horizontalalignment='center')
    plt.show()


def build_spanning_tree(graph_adj, tree_adj, N):
    tree_connection = [0 for i in range(N)]
    build_tree_i = 1
    min_weight_i = 0
    min_weight_j = 1
    min_weight = graph_adj[min_weight_i][min_weight_j]
    for i in range(N):
        for j in range(i + 1, N):
            if min_weight > graph_adj[i][j]:
                min_weight = graph_adj[i][j]
                min_weight_i, min_weight_j = i, j
    tree_adj[min_weight_i][min_weight_j] = min_weight
    tree_adj[min_weight_j][min_weight_i] = tree_adj[min_weight_i][min_weight_j]
    tree_connection[min_weight_i] = tree_connection[min_weight_j] = 1
    print("Build spanning tree, iteration № {}".format(build_tree_i))
    print_matrix(tree_adj)
    print("Connected peaks: {}".format(tree_connection))

    while 0 in tree_connection:
        build_tree_i += 1
        min_weight = None
        min_weight_i = None
        min_weight_j = None
        for i in range(N):
            if tree_connection[i] == 1:
                for j in range(N):
                    if i == j or tree_connection[j] == 1 or tree_adj[i][j] != 0:
                        continue
                    if min_weight is None or min_weight > graph_adj[i][j]:
                        min_weight = graph_adj[i][j]
                        min_weight_i, min_weight_j = i, j
        tree_adj[min_weight_i][min_weight_j] = min_weight
        tree_adj[min_weight_j][min_weight_i] = tree_adj[min_weight_i][min_weight_j]
        tree_connection[min_weight_i] = tree_connection[min_weight_j] = 1
        print("Build spanning tree, iteration № {}".format(build_tree_i))
        print_matrix(tree_adj)
        print("Connected peaks: {}".format(tree_connection))


def clustering_algorithm(graph_adj, N, K):
    for c in range(K - 1):
        max_weight = None
        max_weight_i = None
        max_weight_j = None
        for i in range(N):
            for j in range(i + 1, N):
                if graph_adj[i][j] != 0 and (max_weight is None or graph_adj[i][j] > max_weight):
                    max_weight = graph_adj[i][j]
                    max_weight_i = i
                    max_weight_j = j
        graph_adj[max_weight_i][max_weight_j] = graph_adj[max_weight_j][max_weight_i] = 0


def cluster_graph(graph_adj, N):
    clustered_peaks = [-1 for i in range(N)]
    last_cluster_number = 1
    peaks_to_visit = [0]
    while len(peaks_to_visit) > 0:
        current_vertex = peaks_to_visit.pop()
        current_cluster_number = clustered_peaks[current_vertex]
        adj_peaks = []
        for j in range(N):
            if graph_adj[current_vertex][j] != 0:
                adj_peaks.append(j)
                if clustered_peaks[j] != -1:
                    current_cluster_number = clustered_peaks[j]

        if current_cluster_number == -1:
            current_cluster_number = last_cluster_number
            last_cluster_number += 1

        clustered_peaks[current_vertex] = current_cluster_number
        for adj_vertex in adj_peaks:
            if clustered_peaks[adj_vertex] == -1:
                peaks_to_visit.append(adj_vertex)
                clustered_peaks[adj_vertex] = current_cluster_number
            elif adj_vertex in peaks_to_visit:
                peaks_to_visit.remove(adj_vertex)

        if len(peaks_to_visit) == 0:
            for i in range(N):
                if clustered_peaks[i] == -1:
                    peaks_to_visit.append(i)
                    break
    return clustered_peaks

if __name__ == '__main__':
    N = 10
    K = 3
    graph_adj = [[0 for j in range(N)] for i in range(N)]
    tree_adj = [[0 for j in range(N)] for i in range(N)]

    for i in range(N):
        for j in range(i + 1, N):
            graph_adj[i][j] = np.random.randint(1, 100)
            graph_adj[j][i] = graph_adj[i][j]
    print('Input graph')
    print_matrix(graph_adj)
    draw_graph(graph_adj, N)

    build_spanning_tree(graph_adj, tree_adj, N)
    print('Spanning tree graph')
    print_matrix(tree_adj)
    draw_graph(tree_adj, N)

    clustering_algorithm(tree_adj, N, K)
    print('After cluster highlighting')
    print_matrix(tree_adj)
    draw_graph(tree_adj, N)

    clustered_peaks = cluster_graph(tree_adj, N)
    print('Cluster result')
    print(clustered_peaks)
    draw_graph(tree_adj, N, clustered_peaks)