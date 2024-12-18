import pickle
import time
from absl import app, flags
from tqdm import tqdm
from graphwave.graphwave import *
from cogdl.models.emb.netsmf import NetSMF
import torch
from cogdl.data import Graph

# flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('cg_emb_dim', 80, 'dim')
flags.DEFINE_integer('max_seq', 100, 'Max length of cascade sequence.')
flags.DEFINE_integer('num_s', 2, 'Number of s for spectral graph wavelets.')
flags.DEFINE_integer('observation_time', 1095, 'Observation time.')
# flags.DEFINE_integer('observation_time', 86400, 'Observation time.')
# paths
flags.DEFINE_string('data', 'data/aps/', 'Dataset path.')
# flags.DEFINE_string('data', 'data/twitter/', 'Dataset path.')


def sequence2list(filename, gg):
    graphs = dict()
    with open(filename, 'r') as f:
        for line in f:
            paths = line.strip().split('\t')[:-1][:FLAGS.max_seq + 1]
            graphs[paths[0]] = list()
            for i in range(1, len(paths)):
                nodes = paths[i].split(':')[0]
                time = paths[i].split(':')[1]
                graphs[paths[0]].append([[int(x) for x in nodes.split(',')], int(time)])
                nodes = nodes.split(',')
                if len(nodes) < 2:
                    gg.add_node(nodes[-1])
                else:
                    gg.add_edge(nodes[-1], nodes[-2])
    return graphs, gg


def read_labels(filename):
    labels = dict()
    with open(filename, 'r') as f:
        for line in f:
            id = line.strip().split('\t')[0]
            labels[id] = line.strip().split('\t')[-1]
    return labels


def write_cascade(graphs, labels, filename, gg_emb, id2row, weight=True):
    """
    Input: cascade graphs, global embeddings
    Output: cascade embeddings, with global embeddings appended
    """
    y_data = list()
    id_data = list()
    cascade_input = list()
    cascade_i = 0
    cascade_size = len(graphs)
    total_time = 0

    # for each cascade graph, generate its embeddings via wavelets
    for key, graph in tqdm(graphs.items()):
        start_time = time.time()
        y = int(labels[key])

        # lists for saving embeddings
        cascade_temp = list()
        t_temp = list()

        # build graph
        g = nx.Graph()
        nodes_index = list()
        list_edge = list()
        cascade_embedding = list()
        global_embedding = list()
        t_o = FLAGS.observation_time

        # add edges into graph
        for path in graph:
            t = path[1]
            if t >= t_o:
                continue
            nodes = path[0]
            if len(nodes) == 1:
                nodes_index.extend(nodes)
                t_temp.append(0)
                continue
            else:
                nodes_index.extend([nodes[-1]])
            if weight:
                edge = (nodes[-1], nodes[-2], (1 - t / t_o))  # weighted edge
            else:
                edge = (nodes[-1], nodes[-2])
            list_edge.append(edge)
            t_temp.append(t / t_o)

        if weight:
            g.add_weighted_edges_from(list_edge)
        else:
            g.add_edges_from(list_edge)

        # this list is used to make sure the node order of `chi` is same to node order of `cascade`
        nodes_index_unique = list(set(nodes_index))
        nodes_index_unique.sort(key=nodes_index.index)

        # embedding dim check
        d = FLAGS.cg_emb_dim / (4 * FLAGS.num_s)
        if FLAGS.cg_emb_dim % 4 != 0:
            raise ValueError

        # generate cascade embeddings
        if len(nodes_index_unique) <= 1:
            g.add_node(nodes_index_unique[0])
            chi, _, _ = graphwave_alg(g, np.linspace(0, 100, int(d)),
                                  taus='auto', verbose=False,
                                  nodes_index=nodes_index_unique,
                                  nb_filters=FLAGS.num_s)
        else:
            chi, _, _ = graphwave_alg(g, np.linspace(0, 100, int(d)),
                                      taus='auto', verbose=False,
                                      nodes_index=nodes_index_unique,
                                      nb_filters=FLAGS.num_s)

        # save embeddings into list
        for node in nodes_index:
            cascade_embedding.append(chi[nodes_index_unique.index(node)])
            global_embedding.append(gg_emb[id2row[node]])

        # concat node features to node embedding
        if weight:
            cascade_embedding = np.concatenate([np.reshape(t_temp, (-1, 1)),
                                                np.array(cascade_embedding)[:, :],
                                               np.array(global_embedding)[:, :]],
                                               axis=1)

        # save embeddings
        cascade_temp.extend(cascade_embedding)
        cascade_input.append(cascade_temp)

        # save labels
        y_data.append(y)
        id_data.append(key)

        total_time += time.time() - start_time
        cascade_i += 1
        if cascade_i % 1000 == 0:
            speed = total_time / cascade_i
            eta = (cascade_size - cascade_i) * speed
            print('{}/{}, eta: {:.2f} mins'.format(
                cascade_i, cascade_size, eta/60))

    # write concatenated embeddings into file
    with open(filename, 'wb') as f:
        pickle.dump((cascade_input, y_data, id_data), f)


def get_global_index_and_embedding(gg):
    ids = [int(id) for id in gg.nodes()]
    id2row = dict()
    edges = []
    i = 0
    for id in ids:
        id2row[id] = i
        i += 1
    for edge in gg.edges():
        node1, node2 = id2row[int(edge[0])], id2row[int(edge[1])]
        edges.append([node1, node2])

    tensor_edges = torch.tensor(edges).t()
    graph = Graph(edge_index=tensor_edges)
    print(f"nodes:  {len(gg.nodes())} ; edges: {len(gg.edges())}")
    model = NetSMF(int(FLAGS.cg_emb_dim / 2), 10, 1, 100, 10)
    gg_emb = model.forward(graph)

    return gg_emb, id2row


def main(argv):
    time_start = time.time()

    gg = nx.Graph()

    # get the information of nodes/users of cascades
    graph_train, gg = sequence2list(FLAGS.data + f'train_{FLAGS.observation_time}.txt', gg)
    graph_val, gg = sequence2list(FLAGS.data + f'val_{FLAGS.observation_time}.txt', gg)
    graph_test, gg = sequence2list(FLAGS.data + f'test_{FLAGS.observation_time}.txt', gg)

    # get the information of labels of cascades
    label_train = read_labels(FLAGS.data + f'train_{FLAGS.observation_time}.txt')
    label_val = read_labels(FLAGS.data + f'val_{FLAGS.observation_time}.txt')
    label_test = read_labels(FLAGS.data + f'test_{FLAGS.observation_time}.txt')

    gg_emb, id2row = get_global_index_and_embedding(gg)

    print('Processing time: {:.2f}s'.format(time.time()-time_start))
    print('Start writing train set into file.')
    write_cascade(graph_train, label_train, FLAGS.data + f'train_t{FLAGS.observation_time}_s{FLAGS.max_seq}.pkl', gg_emb, id2row)
    print('Start writing val set into file.')
    write_cascade(graph_val, label_val, FLAGS.data + f'val_t{FLAGS.observation_time}_s{FLAGS.max_seq}.pkl', gg_emb, id2row)
    print('Start writing test set into file.')
    write_cascade(graph_test, label_test, FLAGS.data + f'test_t{FLAGS.observation_time}_s{FLAGS.max_seq}.pkl', gg_emb, id2row)
    print('Processing time: {:.2f}s'.format(time.time()-time_start))


if __name__ == '__main__':
    app.run(main)


