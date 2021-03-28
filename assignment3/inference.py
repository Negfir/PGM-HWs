# assignment 3, comp 766-2, winter 2021 (Siamak Ravanbakhsh)
import numpy as np
import pdb
from utils import tensor_mult
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix

class Inference():
    """
        superclass of inference procedures for an Ising model.
        the model is represented using an adjacency matrix adj:
        n x n symmetric adjacency matrix, the diagonal elements are the local potentials:
        p(x) \propto exp(+\sum_{i,j<i} x_i x_j adj[i,j] + \sum_{i} x_i adj[i,i]) for  x_i \in {-1,+1}
    """

    def __init__(self, adj,
                 verbosity=0  # for more info, use higher verbosity level
    ):
        assert np.all(adj == np.transpose(
            adj)), "the adjacency matrix is not symmetric"
        self.adj = adj
        self._verbosity = verbosity

    def get_marginal(
            target=None
    ):
        """
        return the marginal prob. of xi = 1
        for the list of target variables
        --i.e., a vector of marginals p(xi=1)
        """
        pass

    def update_adjacency(self, new_adj):
        self.adj = new_adj


class CliqueTreeInference(Inference):
    """
    implements exact inference using clique-tree
    """

    def __init__(self, adj,
                 verbosity=0,
                 #normalizing messages helps with overflow in large models,
                 normalize_messages=True,

    ):
        Inference.__init__(self, adj, verbosity=verbosity)
        self._normalize_messages = normalize_messages
        self.cliques = None  # a list of lists
        # a 0-1 matrix wehre ctree[i,j] > 0 means cliques[i] and cliques[j] are connected in the clique-tree
        self.ctree_adj = None
        self.chordal = None  # the 0-1 chordal graph
        self.order = None  # the elimination order
        # dictionary identifying the parent of each node in the ctree with root=0
        self.parents = None
        # dictionary identifying the children of each node in the ctree with root=0
        self.children = None
        # dictionary of messages sent from clique i->j (i,j):msg, where the variables in the sepset appear in their natural order in the msg tensor
        self.messages = {}
        # a list of tensors, corresponding to clique potentials (i.e., product of all factors associated with a clique)
        self.clique_potentials = None
        # a dict of tensors, corresponding to the marginal beliefs over cliques
        self.clique_beliefs = {}
        # indicates whether the message updates have been sent or not
        self._calibrated = False
        # build the clique-tree from the adjacency matrix
        self._build_clique_tree()


    def update_adjacency(self, new_adj):
        self.adj = new_adj
        self.ctree_adj = None
        self.order = None
        self.chordal_adj = None
        self.messages = {}
        self.parents = None
        self.children = None
        self.cliques = None
        self.clique_potentials = None
        self.clique_beliefs = None
        self._build_clique_tree()
        self._calibrated = False

    def _min_fill(self):
        """
        return the elimination order (a list of indices) as well
        as the resulting chordal graph baded on min_fill huristic
        chordal_adj[i,j] = 1 iff (i,j) are connected.
        The diagonal of adj should be zero.
        """
        adj = self.adj
        def num_fill_edges(mask,i):
            """
            number of fill edges created by eliminating node i
            in the graph with 0-1 adjacency matrix mask
            """
            n = mask.shape[0]
            nb = np.nonzero(mask[i, :])[0]
            clique_edges = nb.shape[0]*(nb.shape[0]-1)/2
            current_edges = mask[np.ix_(nb, nb)].sum()/2
            return clique_edges - current_edges
        assert np.all(adj == np.transpose(
            adj)), "the adjacency matrix should be symmetric"
        n = adj.shape[0]
        mask = (np.abs(adj) > 0).astype(float)
        np.fill_diagonal(mask, 0)
        order = []
        available = list(range(n))
        total_fill_edges = 0
        for _ in range(n):
            mask_iter = mask[np.ix_(available, available)]
            fills = [num_fill_edges(mask_iter, j)
                     for j in range(len(available))]
            best_ind = np.argmin(fills)
            num_fills = fills[best_ind]
            total_fill_edges += num_fills
            best = available[best_ind]
            neighbors_ind = np.nonzero(mask_iter[best_ind, :])[0]
            neighbors = [available[i] for i in neighbors_ind]
            mask[np.ix_(neighbors, neighbors)] = 1
            mask[neighbors, neighbors] = 0
            available.pop(best_ind)
            order.append(best)
        return order, mask

    def _max_cardinality_search(self, mask):
        """
        mask is the adjacency matrix for 0-1 chordal graph
        this method returns a list of lists: the set of maximal cliques
        we can also return sep-sets here, but we do that using max-spanning-tree later
        """
        n = mask.shape[0]
        cliques = [[]]  # maintains the list of cliques
        last_mark = -1  # number of marked neighbors for prev. node
        marks = [[] for i in range(n)]  # a set tracking the marked neighbors of each node
        mark_size = np.zeros(n)  # number of marked neighbors for each node
        remaining = list(range(n))
        for _ in reversed(range(n)):
            node = remaining[np.argmax(mark_size[remaining])]
            if mark_size[node] <= last_mark:  # moving into a new clique
                cliques.append(marks[node] + [node])
            else:  # add it to the last clique
                cliques[-1].append(node)
            nb_node = np.nonzero(mask[node,:])[0]  # neighbors of node
            for nb in nb_node:  # update the marks for neighbors
                marks[nb].append(node)
                mark_size[nb] += 1
            last_mark = mark_size[node]
            remaining.remove(node)
        sorted_cliques = [sorted(c) for c in cliques]
        return sorted_cliques

    def _get_directed_tree(self, adj, root=0):
        """
        produce a directed tree from the adjacency matrix, with the given root
        return a dictionary of children and parents for each node
        """
        visited = set()
        to_visit = set([0])
        n = adj.shape[0]
        rest = set(range(1, n))
        parents = {root:None}
        children = {}
        while len(to_visit) > 0:
            current = to_visit.pop()
            nexts = set(np.nonzero(adj[current, :])[0]).intersection(rest)
            for j in nexts:
                parents[j] = current
            children[current] = frozenset(nexts)
            to_visit.update(nexts)
            rest.difference_update(nexts)
            visited.add(current)
        assert len(rest) == 0, "the clique tree is disconnected!"
        return parents, children

    def _calc_clique_potentials(self, cliques):
        """
        calculate the potential/factor associated with each clique
        as the product of factors associated with it.
        Note that each local and pairwise factor is associated with
        a single clique (family-preserving property)
        """
        adj = self.adj
        n = adj.shape[0]
        local_set = set(range(n))
        nonz = np.nonzero(adj)
        pairwise_set = set([(nonz[0][i], nonz[1][i]) for i in range(
            len(nonz[0])) if nonz[0][i] < nonz[1][i]])  # the edge-set
        local_tmp_factor = np.ones([2])
        pairwise_tmp_factor = np.ones([2, 2])
        clique_potentials = []
        for cl in cliques:
            cl_vars = sorted(list(cl))
            cl_factor = np.ones(len(cl) * [2])
            for loc in local_set.intersection(cl):
                local_tmp_factor[0] = np.exp(-adj[loc, loc])
                local_tmp_factor[1] = np.exp(adj[loc, loc])
                cl_factor = tensor_mult(cl_factor, local_tmp_factor, [cl_vars.index(loc)], [0])
            # remove the local factors that are already accounted for (family-preserving property)
            local_set.difference_update(cl)
            for i in cl:
                for j in cl:
                    if i < j and (i, j) in pairwise_set:
                        pairwise_tmp_factor[0, 0] = np.exp(adj[i, j])
                        pairwise_tmp_factor[1, 1] = pairwise_tmp_factor[0, 0]
                        pairwise_tmp_factor[0, 1] = np.exp(-adj[i, j])
                        pairwise_tmp_factor[1, 0] = pairwise_tmp_factor[0, 1]
                        cl_factor = tensor_mult(cl_factor, pairwise_tmp_factor,
                                [cl_vars.index(i), cl_vars.index(j)], [0, 1])
                        pairwise_set.remove((i, j))
            clique_potentials.append(cl_factor.copy())
        return clique_potentials

    def _build_clique_tree(self):
        """
        builds the clique-tree from the adjacency matrix by
        1. triangulating the graph to get a chordal graph
        2. find the maximal cliques in the chordal graph
        3. calculating the clique-potentials
        4. selecting the sepsets using max-spanning tree
        5. selecting a root node and building a directed tree
        this method does not calibrate the tree -- i.e., it doesn't
        send the messages upward and downward in the tree
        """
        if self._verbosity > 0:
            print("building the clique-tree ...", flush=True)
        order, chordal_adj = self._min_fill()
        if self._verbosity > 1:
            print("\t found the elimination order {}".format(order), flush=True)
            chordal_viz = chordal_adj + (self.adj != 0)  # so that the color of the chords is different
            np.fill_diagonal(chordal_viz, 0)
        cliques = self._max_cardinality_search(chordal_adj)
        if self._verbosity > 1:
            print("\t number of maximal cliques: {} with max. size: {}".format(len(cliques),
                max([len(c) for c in cliques])), flush=True)
            labels = [[c for c in range(len(cliques)) if i in cliques[c]] for i in range(self.adj.shape[0])]
        if self._verbosity > 1:
            print("\t calculating clique potentials")
        # assign each factor (pairwise or local) to a clique and calculate the clique-potentials
        clique_potentials = self._calc_clique_potentials(cliques)
        # find the size of septsets between all cliques and use max-spanning tree to build the clique-tree
        sepset_size = np.zeros((len(cliques), len(cliques)))
        for i, cl1 in enumerate(cliques):
            for j, cl2 in enumerate(cliques):
                if i != j:
                    sepset_size[i, j] = max(len(set(cl1).intersection(cl2)), .1)
        if self._verbosity > 1:
            print("\t finding the max-spanning tree", flush=True)
        # use scipy for max-spanning-tree
        ctree = minimum_spanning_tree(
            csr_matrix(-sepset_size)).toarray().astype(int)
        # make it symmetric
        ctree_adj = (np.maximum(-np.transpose(ctree.copy()), -ctree) > 0)
        # set the first cluster to be the root and build the directed tree
        root = 0
        parents, children = self._get_directed_tree(ctree_adj, root)
        self.parents = parents
        self.children = children
        self.chordal_adj = chordal_adj
        self.cliques = cliques
        self.ctree_adj = ctree_adj
        self.clique_potentials = clique_potentials
        if self._verbosity > 0:
            print("... done!", flush=True)

    def _calc_message(self,
                      src_node,  # source of the message
                      dst_set,  # a set of destinations,
                      upward,
    ):
        """
        if the downward, the message to all destinations is calculated by first
        obtaining the belief and dividing out the corresponding incoming messages
        This assumes that the distribution is positive and therefore messages are never zero.
        during the upward pass there is only a single destination and the message is obtained directly
        should also work when the dst_set is empty (producing belief over the leaf nodes)
        """
        # incoming messages are from these clusters
        incoming = set(self.children[src_node])
        if self.parents[src_node] is not None:
            incoming.add(self.parents[src_node])
        if upward:
            incoming.difference_update(dst_set)  # only has one destination
            assert len(dst_set) == 1, "should have a single receiver in the upward pass!"
        factor = self.clique_potentials[src_node].copy()
        clique_vars = self.cliques[src_node]
        for r in incoming:
            sepset = list(set(self.cliques[r]).intersection(set(clique_vars)))
            # find the index of sepset in the clique potential
            inds = sorted([clique_vars.index(i) for i in sepset])
            # multiply with the incoming message from the child
            factor = tensor_mult(factor, self.messages[(r,src_node)], inds, list(range(len(sepset))))
        for dst_node in dst_set:
            tmp_factor = factor.copy()
            if not upward:  # divide out the incoming message to produce the outgoing message
                sepset = set(self.cliques[dst_node]).intersection(set(clique_vars))
                # find the index of sepset in the clique potential
                inds = sorted([clique_vars.index(i) for i in sepset])
                # multiply with the incoming message from the child
                tmp_factor = tensor_mult(tmp_factor, 1./self.messages[(dst_node,src_node)], inds, list(range(len(sepset))))
            outgoing_vars = set(clique_vars).intersection(set(self.cliques[dst_node]))
            sum_over_vars = set(clique_vars) - set(outgoing_vars)
            sum_over_vars_inds = sorted([clique_vars.index(i) for i in sum_over_vars])
            msg = np.sum(tmp_factor, axis=tuple(sum_over_vars_inds))
            if self._normalize_messages:
                msg /= np.sum(msg)
            self.messages[(src_node,dst_node)] = msg
            if self._verbosity > 2:
                print("{} -> ({})-> {}".format(clique_vars, outgoing_vars ,self.cliques[dst_node]), flush=True)
        return factor  # is used to set the clique-marginals in the downward pass

    def _upward(self, root=0):
        """
        send the message from leaf nodes towards the root
        each node sends its message as soon as received messages from its children
        """
        if self._verbosity > 0:
            print("sending messages towards the root node", end="", flush=True)
        # leaf nodes
        ready_to_send = set([node for node, kids in self.children.items() if len(kids) == 0])
        #until root receives all its incoming messages
        while root not in ready_to_send:
            if self._verbosity > 0:
                print(".", end="", flush=True)
            current = ready_to_send.pop()
            # send the message to the parent
            parent = self.parents[current]
            self._calc_message(current, {parent}, True)
            #if the parent has received all its incoming messages, add it to ready_to_send
            parent_is_ready = np.all([((ch,parent) in self.messages.keys()) for ch in self.children[parent]])
            if parent_is_ready: ready_to_send.add(parent)
        if self._verbosity > 0:
            print("", end="\n", flush=True)

    def _downward(self, root=0):
        """
        send the messages downward from the root
        each node sends its message to its children as soon as received messages from its parent
        """

        if self._verbosity > 0:
            print("sending messages towards the leaf nodes", end="", flush=True)
        ready_to_send = set([root])
        while len(ready_to_send) > 0:
            current = ready_to_send.pop()
            self.clique_beliefs[current] = self._calc_message(current, self.children[current], False)
            ready_to_send.update(self.children[current])
            if self._verbosity > 0:
                print(".", end="", flush=True)
        if self._verbosity > 0:
            print("", end="\n", flush=True)

    def get_marginal(self, target):
        """
        return the marginal prob. of xi = 1
        for the list of target variables
        --i.e., a vector of marginals p(xi=1)
        these are calculated using clique_beliefs
        """
        if not self._calibrated:
            self._upward()
            self._downward()
            self._calibrated = True

        if self._verbosity > 0:
            print("calculating the marginals for {} target variables".format(len(target)), flush=True)

        target_set = set(target)
        p1 = {}
        for c, clique in enumerate(self.cliques):
            cl_var_set = set(clique)
            for v in target_set.intersection(cl_var_set):
                v_ind = clique.index(v)
                summation_inds = list(set(range(len(cl_var_set))).difference({v_ind}))
                mrg = np.sum(self.clique_beliefs[c], axis=tuple(summation_inds))

                mrg /= np.sum(mrg)
                p1[v] = mrg[1]
        p1_arr = np.array([p1[v] for v in target])
        return p1_arr
