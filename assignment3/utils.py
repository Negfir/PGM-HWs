import numpy as np
import networkx as nx
import pylab as plt
import pdb



def tensor_mult(a, # n_1 x n_2 x ... x n_d tensor
               b, # m_{1} x m_{2} x ... x m_{l} tensor
               a_dims, # list of dimensions of a to broadcast multiply
               b_dims, # list of dimensions of b to broadcast multiply
):
    """
    multiplies the two tensors along the specified dimensions. a and b should have the same size in these dimensions.
    the result is a tensor c with dimensions equal to a.ndim + b.ndim - len(a_dims).
    the shape of the first a.ndim dimensions of c is the same as a, while the remaining dimensions match the dimensions of 
    b that did not participate in multiplication.

    example: 
    a = np.ones([2,2,3,5,4])
    b = np.random.rand(7,4,3,2)
    c = tensor_mult(a, b, [1,4], [3,1])
    c.shape # [2,2,3,5,4,7,3] 
    """
    
    assert len(a_dims) == len(b_dims), "a_dims and b_dims should have the same length!"
    assert np.all([a.shape[a_dims[i]] == b.shape[b_dims[i]] for i in range(len(a_dims))]), "a_dims %s and b_dims%s dimensions do not match!" %(a_dims, b_dims)

    d_a = a.ndim
    d_b = b.ndim
    #bring the relevant dimensions to the front
    missing_a = [i for i in range(d_a) if i not in a_dims]
    new_order_a = a_dims + missing_a
    a_t = np.transpose(a, tuple(new_order_a))
    missing_b = [i for i in range(d_b) if i not in b_dims]
    new_order_b = b_dims + missing_b
    b_t = np.transpose(b, tuple(new_order_b))

    #expand the tensors to make the shapes compatible
    a_t = np.reshape(a_t, list(a_t.shape)+len(missing_b)*[1])
    b_t = np.reshape(b_t, [b.shape[i] for i in b_dims]+len(missing_a)*[1]+[b.shape[i] for i in missing_b])

    #multiply
    c_t = a_t * b_t

    #reshape the results: a_dims ; missing_a ; missing_b -> original shape of a ; missing_b
    a_t_index = np.unique(new_order_a, return_index=True)[1].tolist()
    b_t_index = np.arange(d_a, d_a+d_b-len(a_dims)).tolist()
    c = np.transpose(c_t, a_t_index+b_t_index)
    return c


def draw_graph(adj=None, G = None, marginals=None,
               draw_edge_color=False, title=None,
               node_size=300, node_labels=None):

    node_color = marginals
    if G is None:
        assert adj is not None, "you have to provide either the adjacency matrix or the graph"        
        G = nx.from_numpy_array(adj)
    edge_color = G.number_of_edges()*[1]
    n = G.number_of_nodes()
    if adj is not None:
        edges = adj[np.triu_indices(n,1)]  # strict upper triangle inds
        if draw_edge_color:
            edge_color = edges[edges != 0].ravel().astype(float).tolist()
    if node_labels is not None:
        node_dict = dict([(i, str(node_labels[i])) for i in range(n)])
    else: node_dict = None
    nx.draw(G, node_color=marginals, edge_color = edge_color,
                     label=title, node_size = node_size,
                     labels=node_dict)
    plt.show()



def logistic(x): #clipping for numerical stability
    return 1. / (1. + np.exp(-np.clip(x,-100,100)))

