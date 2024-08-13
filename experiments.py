import numpy as np
import pdb
import pickle

from abc import ABCMeta, abstractmethod

#It is okay to use n^2 memory
class Graph:
    def __init__(self, mat):
        self.matrix_ = mat

    def get_size(self):
        return self.matrix_.shape[0]

    def get_adj_matrix(self):
        return self.matrix_
    
    def get_adj_list(self, i):
        return self.matrix_[i, :]
    
    def get_degree_vec(self):
        return self.matrix_.sum(axis=1).astype('int')

    def get_dim(self):
        return self.matrix_.shape[0]

    def matrix_from_snap_graph(self, graph):
        n = graph.GetNodes()
        mat = np.zeros((n,n), dtype='<u1')
        for E in graph.Edges():
            e1 = E.GetSrcNId()
            e2 = E.GetDstNId()
            mat[e1][e2] = 1
            mat[e2][e1] = 1
        return mat

    def get_noisy_graph(self, rho):
        n = self.get_dim()
        RR = np.random.choice([0,1], (n,n), p=[1-rho, rho])
        new_mat = (RR + self.matrix_) % 2
        for i in range(0, n):
            new_mat[i,i] = 0
        return Graph(new_mat)
        
class AdjacencyListGraph(Graph):
    def __init__(self, n, file_name):
        mat = np.zeros((n,n), dtype='<u1')
        with open(file_name) as f:
            for line in f:
                nums = line.split(' ')
                assert(len(nums) == 2)
                a = int(nums[0])
                b = int(nums[1])
                mat[a][b] = 1
                mat[b][a] = 1
        self.matrix_ = mat

class GNPGraph(Graph):
    def __init__(self, n, p):
        mat = np.random.choice(np.array([0,1], dtype='<u1'), size=(n, n), p =
                               (p, 1-p))
        for i in range(0, n):
            mat[i, i:] = mat[i:, i]
            mat[i, i] = 0
        self.matrix_ = mat

class GraphTester:
    @staticmethod
    def test_gnm_graph(n, p):
        g = GNPGraph(n, p)
        M = g.get_adj_matrix()
        print("Adjacency mat. of regular graph")
        for i in range(0, n):
            for j in range(0, n):
                print(M[i,j], end='')
            print()
        
        deg = g.get_degree_vec()
        print("Degree vec of graph")
        for i in range(0, n):
            print(deg[i], end='')
        print()

        print("Adjacency mat. of noisy graph")
        rho = 1/3.0
        h = g.get_noisy_graph(rho)
        M = h.get_adj_matrix()
        for i in range(0, n):
            for j in range(0, n):
                print(M[i,j], end='')
            print()

    @staticmethod
    def test_facebook_graph():
        n = 4039
        g = AdjacencyListGraph(n, 'graphs/facebook_combined.txt')
        M = g.get_adj_matrix()
        print("First 100 rows of adjacency matrix")
        for i in range(0, 100):
            for j in range(0, 100):
                print(M[i,j], end='')
            print()

        deg = g.get_degree_vec()
        print("Degree vec of graph")
        for i in range(0, n):
            print(deg[i], end='')
        print()

#Base class running manipulation attacks
class Manipulation:
    def __init__(self, input_graph, n_mal, epsilon, delta):
        __metaclass__ = ABCMeta
        self.input_graph = input_graph
        self.max_num_mal = n_mal
        self.epsilon = epsilon
        self.delta = delta
        self.rho = 1 / (1 + np.exp(epsilon))
        self.n = self.input_graph.get_dim()
        self.attacks = []
        self.available_players = list(range(0, self.n))
        self.num_mal = 0

    def add_attacks(self, attack_desc):
        num_mal = 0
        players = self.available_players.copy()
        attacks = []
        for i in range(0, len(attack_desc)):
            mal_targ, hon_targ, attack_size, method = attack_desc[i]
            num_mal += mal_targ + attack_size
            if isinstance(method, str):
                if method == 'Random':
                    selected = np.random.choice(players, mal_targ + hon_targ + attack_size, replace=False)
                    players = np.setdiff1d(players, selected)
                    mid = mal_targ + hon_targ
                    attacks.append((selected[:mal_targ], selected[mal_targ:mid], selected[mid:]))
                elif method == 'Friends':
                    index = np.random.choice(players)
                    mal_users = np.where(self.input_graph.get_adj_list(index) == 1)[0]
                    if len(mal_users) >= attack_size:
                        mal_users = np.random.choice(mal_users, attack_size, replace=False)
                        players = np.setdiff1d(players, [index] + list(mal_users))
                    else:
                        players = np.setdiff1d(players, [index] + list(mal_users))
                        add_mal = np.random.choice(players, attack_size - len(mal_users), replace=False)
                        mal_users = np.concatenate((mal_users, add_mal))
                        players = np.setdiff1d(players, add_mal)
                    
                    targets = np.random.choice(players, mal_targ + hon_targ - 1, replace=False)
                    if hon_targ > 0:
                        attacks.append((list(targets[:mal_targ]), [index] + list(targets[mal_targ:]), mal_users))
                    else:
                        attacks.append(([index] + list(targets), [], mal_users))
                else: 
                    raise Exception('Method %s not understood' % method)
            elif isinstance(method, list):
                candidates = np.intersect1d(method, players)
                num_selected = hon_targ + mal_targ + attack_size
                if num_selected > len(candidates):
                    raise Exception("Too few users in community")
                    num_selected = len(candidates)
                selected = np.random.choice(candidates, num_selected, replace=False)
                players = np.setdiff1d(players, selected)
                mid = min(mal_targ + hon_targ, len(candidates))
                attacks.append((selected[:mal_targ], selected[mal_targ:mid], selected[mid:]))
            else:
                raise Exception('Method not understood')

        tot_num_mal = num_mal + self.num_mal
        if tot_num_mal > self.max_num_mal:
            raise Exception('Too many malicious users added. Attacks not added')
        else:
            self.num_mal = tot_num_mal
            self.available_players = players
            self.attacks += attacks

    def reset_attacks(self):
        self.attacks = []
        self.available_players = list(range(0, self.n))
        self.num_mal = 0

    def set_attacks_like(self, other):
        self.attacks = other.attacks.copy()
        self.available_players = other.available_players.copy()
        self.num_mal = other.num_mal

    def set_epsilon(self, new_epsilon):
        self.epsilon = new_epsilon
        self.rho = 1 / (1 + np.exp(new_epsilon))

    def get_input_graph(self):
        return self.input_graph
    
    def get_num_players(self):
        return self.n

    @abstractmethod
    def get_degree_estimates(self):
        pass

    def get_results(self, verbose=True):
        D1 = self.get_degree_estimates()
        D2 = self.input_graph.get_degree_vec()
        target_diffs = []
        target_num_na = 0
        for a in self.attacks:
            for i in list(a[0]):
                if np.isnan(D1[i]):
                    target_num_na += 1
                else:
                    target_diffs.append(abs(D1[i] - D2[i]))

        honest_target_diffs = []
        honest_target_num_na = 0
        for a in self.attacks:
            for i in list(a[1]):
                if np.isnan(D1[i]):
                    honest_target_num_na += 1
                else:
                    honest_target_diffs.append(abs(D1[i] - D2[i]))
        mal_num_na = 0
        for a in self.attacks:
            for i in list(a[2]):
                mal_num_na += np.isnan(D1[i])

        honest_diffs = []
        honest_num_na = 0
        for i in self.available_players:
            if np.isnan(D1[i]):
                honest_num_na += 1
            else:
                honest_diffs.append(abs(D1[i] - D2[i]))

        td = np.array(target_diffs)
        td_stats = (0,0,0)
        htd_stats = (0,0,0)
        hd_stats = (0,0,0)
        if len(target_diffs) > 0:
            td_stats = (td.max(), np.linalg.norm(td), td.mean())
        htd = np.array(honest_target_diffs)
        if len(honest_target_diffs) > 0:
            htd_stats = (htd.max(), np.linalg.norm(htd), htd.mean())
        hd = np.array(honest_diffs)
        if len(honest_diffs) > 0:
            hd_stats = (hd.max(), np.linalg.norm(hd), hd.mean())

        if verbose:
            print('Malicious Target Error:')
            if len(target_diffs) > 0:
                print('\t%0.2f (max), %0.2f (l_2), %0.2f (mean)' % td_stats)
            print("\t%d/%d Disqualified" % (target_num_na, len(target_diffs) + target_num_na))
            
            print('Honest Target Error:')
            if len(honest_target_diffs) > 0:
                print('\t%0.2f (max), %0.2f (l_2), %0.2f (mean)' % htd_stats)
            print("\t%d/%d Disqualified" % (honest_target_num_na, len(honest_target_diffs) + honest_target_num_na))
            
            print('Honest Error:')
            if len(honest_diffs) > 0:
                print('\t%0.2f (max), %0.2f (l_2), %0.2f (mean)' % hd_stats)
            print("\t%d/%d Disqualified" % (honest_num_na, len(honest_diffs) + honest_num_na))
        return (td_stats, (target_num_na, len(target_diffs) + target_num_na), mal_num_na,
                htd_stats, (honest_target_num_na, len(honest_target_diffs) +
                honest_target_num_na), hd_stats, (honest_num_na, len(honest_diffs)
                + honest_num_na))

class ResponseLaplaceManipulation(Manipulation):
    def __init__(self, input_graph, n_mal, epsilon, delta,
                 laplace_eps_frac, inflation_factors, threshold_reduction = None):
        super(ResponseLaplaceManipulation, self).__init__(
                input_graph, n_mal, epsilon, delta)
        self.laplace_eps_frac = laplace_eps_frac
        self.set_epsilon(epsilon)
        if threshold_reduction is None:
            self.threshold_reduction = [1,1,1]
        else:
            self.threshold_reduction = threshold_reduction
        self.inflation_factors = inflation_factors

    def set_epsilon(self, new_epsilon):
        self.epsilon = new_epsilon
        rr_eps = new_epsilon * (1 - self.laplace_eps_frac)
        lap_eps = new_epsilon - rr_eps
        self.epsilon_divy = (rr_eps, lap_eps)
        self.rho = 1 / (1 + np.exp(rr_eps))
    
    def construct_response(self, m_i, mal_users):
        thresh = self.get_thresh()
        rho = self.rho
        C = self.inflation_factors
        input_graph = self.input_graph.get_adj_matrix()[m_i, :].copy()
        RR = np.random.choice([0,1], input_graph.shape, p=[1-rho, rho])
        response = (RR + input_graph) % 2
        response[mal_users] = 1
        response[m_i] = 1
        hon_0s = np.where(response == 0)[0]
        connect_0s = min( int(C[0] * hon_0s.size), hon_0s.size)
        response[hon_0s[:connect_0s]] = 1
        response[m_i] = 0
        RR = np.random.choice([0,1], input_graph.shape, p=[1-rho, rho])
        response2 = (RR + input_graph) % 2
        response2[mal_users] = 1
        er_11 = (response * response2).sum()
        ed_rr = (er_11 - rho**2 * self.n) / (1-2*rho)
        cheat_deg = ed_rr + 2 * C[1] * thresh / (1-2*rho)
        return (response, cheat_deg)

    def get_thresh(self):
        T = self.threshold_reduction
        i1 = self.max_num_mal + T[1] * np.sqrt(2 * self.rho * self.n * np.log(8 * self.n / self.delta))
        return T[0] * i1

    def run_attack(self, attack, resp, deg_resp):
        mal_targets, hon_targets, mal_users = attack
        for mal_user in mal_users:
            resp[mal_user, hon_targets] = 0
            resp[mal_user, mal_targets] = 1
        
        for mal_user in mal_targets:
            cheat_resp, cheat_deg = self.construct_response(mal_user, mal_users)
            resp[mal_user, :] = cheat_resp
            deg_resp[mal_user] = cheat_deg

        return (resp, deg_resp)


    def get_degree_estimates(self):
        degs = self.input_graph.get_degree_vec()
        mat = self.input_graph.get_adj_matrix().copy()

        g_new = Graph(mat).get_noisy_graph(self.rho)
        noisy_resp = g_new.get_adj_matrix()
        noise = np.random.laplace(1.0/self.epsilon_divy[1], size =
                self.n)
        priv_degs = degs + noise

        #Poison the responses
        for a in self.attacks:
            noisy_resp, priv_degs = self.run_attack(a, noisy_resp, priv_degs)

        thresh = self.get_thresh()
        self.set_epsilon(self.epsilon)
        deg_vec = np.zeros_like(priv_degs)
        for i in range(0, self.n):
            my_responses = noisy_resp[i, :]
            other_responses = noisy_resp[:, i]
            r_11 = (my_responses * other_responses).sum()
            r_01 = ((1-my_responses) * other_responses).sum()
            if (np.abs(r_01 - (self.n - 1) * self.rho *
                (1-self.rho))  >= thresh):
                deg_vec[i] = np.nan
                #print('rejected 1')
                continue
            rr_deg_est = 1.0 / (1-2 * self.rho) * \
                (r_11 - (self.n - 1) * self.rho**2)
            if (np.abs(rr_deg_est - priv_degs[i]) >= 2 * thresh / (1-2 * self.rho)
                + self.threshold_reduction[2] * np.log(2 * self.n / self.delta) / self.epsilon_divy[1]):
                deg_vec[i] = np.nan
                #print('rejected Lap: %0.2f %0.2f' % (rr_deg_est, priv_degs[i]))
            else:
                deg_vec[i] = priv_degs[i]
        #pdb.set_trace()
        return deg_vec

class InputLaplaceManipulation(Manipulation):
    def __init__(self, input_graph, n_mal, epsilon, delta,
                 laplace_eps_frac, inflation_factors, threshold_reduction = None):
        super(InputLaplaceManipulation, self).__init__(
                input_graph, n_mal, epsilon, delta)
        self.laplace_eps_frac = laplace_eps_frac
        self.set_epsilon(epsilon)
        self.inflation_factors = inflation_factors
        if threshold_reduction is None:
            self.threshold_reduction = [1,1,1]
        else:
            self.threshold_reduction = threshold_reduction

    def set_epsilon(self, new_epsilon):
        self.epsilon = new_epsilon
        rr_eps = new_epsilon * (1 - self.laplace_eps_frac)
        lap_eps = new_epsilon - rr_eps
        self.epsilon_divy = (rr_eps, lap_eps)
        self.rho = 1 / (1 + np.exp(rr_eps))

    def construct_response(self, m_i, mal_users):
        thresh = self.get_thresh()
        rho = self.rho
        C  = self.inflation_factors
        input_graph = self.input_graph.get_adj_matrix()[m_i, :].copy()
        input_pois = input_graph.copy()
        input_pois[m_i] = 1
        hon_0s = np.where(input_pois == 0)[0]
        connect_0s = min( int(C[0] * hon_0s.size), hon_0s.size)
        input_pois[hon_0s[:connect_0s]] = 1
        input_pois[m_i] = 0
        RR = np.random.choice([0,1], input_graph.shape, p=[1-rho, rho])
        e_resp1 = (RR + input_graph) % 2
        RR = np.random.choice([0,1], input_graph.shape, p=[1-rho, rho])
        e_resp2 = (RR + input_pois) % 2
        er_11 = (e_resp1 * e_resp2).sum()
        ed_rr = (er_11 - rho**2 * self.n) / (1-2*rho)
        cheat_deg = ed_rr + 2 * C[1] * thresh / (1-2*rho)
        return (input_pois, cheat_deg)

    def get_thresh(self):
        T = self.threshold_reduction
        i1 = self.max_num_mal * (1-2*self.rho) + T[1] * np.sqrt(8 * \
                                            max(self.n * self.rho,
                                                self.max_num_mal) *
                                              np.log(8 * self.n / self.delta))
        return T[0] * i1

    def run_attack(self, attack, resp, deg_resp):
        mal_targets, hon_targets, mal_users = attack
        for mal_user in mal_users:
            resp[mal_user, hon_targets] = 0
            resp[mal_user, mal_targets] = 1
        
        for mal_user in mal_targets:
            cheat_resp, cheat_deg = self.construct_response(mal_user, mal_users)
            resp[mal_user, :] = cheat_resp
            deg_resp[mal_user] = cheat_deg

        return (resp, deg_resp)

    def get_degree_estimates(self):
        degs = self.input_graph.get_degree_vec()
        mat = self.input_graph.get_adj_matrix().copy()

        #Poison the responses
        for a in self.attacks:
            mat, degs = self.run_attack(a, mat, degs) 

        g_new = Graph(mat).get_noisy_graph(self.rho)
        noisy_resp = g_new.get_adj_matrix()
        noise = np.random.laplace(1.0/self.epsilon_divy[1], size =
                self.n)
        priv_degs = degs + noise

        thresh = self.get_thresh()
        self.set_epsilon(self.epsilon)
        deg_vec = np.zeros_like(priv_degs)
        for i in range(0, self.n):
            my_responses = noisy_resp[i, :]
            other_responses = noisy_resp[:, i]
            r_11 = (my_responses * other_responses).sum()
            r_01 = ((1-my_responses) * other_responses).sum()
            if (np.abs(r_01 - (self.n - 1) * self.rho *
                (1-self.rho))  >= thresh):
                deg_vec[i] = np.nan
                #print('rejected 1')
                continue
            rr_deg_est = 1.0 / (1-2 * self.rho) * \
                (r_11 - (self.n - 1) * self.rho**2)
            if (np.abs(rr_deg_est - priv_degs[i]) >= 2 * thresh / (1-2 * self.rho)
                + self.threshold_reduction[2] * np.log(2 * self.n / self.delta) / self.epsilon_divy[1]):
                deg_vec[i] = np.nan
                print('rejected Lap: %0.2f %0.2f' % (rr_deg_est, priv_degs[i]))
            else:
                deg_vec[i] = priv_degs[i]
        #pdb.set_trace()
        return deg_vec

class ResponseRRManipulation(Manipulation):
    def __init__(self, input_graph, n_mal, epsilon, delta, inflation_factors, 
                 threshold_reduction = None):
        super(ResponseRRManipulation, self).__init__(
                input_graph, n_mal, epsilon, delta)
        self.set_epsilon(epsilon)
        if threshold_reduction is None:
            self.threshold_reduction = [1,1]
        else:
            self.threshold_reduction = threshold_reduction
        self.inflation_factors = inflation_factors

    def set_epsilon(self, new_epsilon):
        self.epsilon = new_epsilon
        self.rho = 1 / (1 + np.exp(new_epsilon))
    
    def construct_response(self, m_i, mal_users):
        rho = self.rho
        C = self.inflation_factors
        input_graph = self.input_graph.get_adj_matrix()[m_i, :].copy()
        RR = np.random.choice([0,1], input_graph.shape, p=[1-rho, rho])
        response = (RR + input_graph) % 2
        response[mal_users] = 1
        response[m_i] = 1
        hon_0s = np.where(response == 0)[0]
        connect_0s = min( int(C[0] * hon_0s.size), hon_0s.size)
        response[hon_0s[:connect_0s]] = 1
        response[m_i] = 0
        return response

    def get_thresh(self):
        T = self.threshold_reduction
        i1 = self.max_num_mal + T[1] * np.sqrt(2 * self.rho * self.n * np.log(4 * self.n / self.delta))
        return T[0] * i1

    def run_attack(self, attack, resp):
        mal_targets, hon_targets, mal_users = attack
        for mal_user in mal_users:
            resp[mal_user, hon_targets] = 0
            resp[mal_user, mal_targets] = 1
        
        for mal_user in mal_targets:
            cheat_resp = self.construct_response(mal_user, mal_users)
            resp[mal_user, :] = cheat_resp

        return resp

    def get_degree_estimates(self):
        self.set_epsilon(self.epsilon)
        mat = self.input_graph.get_adj_matrix().copy()

        g_new = Graph(mat).get_noisy_graph(self.rho)
        noisy_resp = g_new.get_adj_matrix()

        #Poison the responses
        for a in self.attacks:
            noisy_resp = self.run_attack(a, noisy_resp)
        thresh = self.get_thresh()
        deg_vec = np.zeros(self.n)
        for i in range(0, self.n):
            my_responses = noisy_resp[i, :]
            other_responses = noisy_resp[:, i]
            r_11 = (my_responses * other_responses).sum()
            r_01 = ((1-my_responses) * other_responses).sum()
            if (np.abs(r_01 - (self.n - 1) * self.rho *
                (1-self.rho))  >= thresh):
                deg_vec[i] = np.nan
                #print('rejected 1')
                continue
            rr_deg_est = 1.0 / (1-2 * self.rho) * \
                (r_11 - (self.n - 1) * self.rho**2)
            deg_vec[i] = rr_deg_est

        #pdb.set_trace()
        return deg_vec

class InputRRManipulation(Manipulation):
    def __init__(self, input_graph, n_mal, epsilon, delta, inflation_factors, 
                 threshold_reduction = None):
        super(InputRRManipulation, self).__init__(
                input_graph, n_mal, epsilon, delta)
        self.set_epsilon(epsilon)
        if threshold_reduction is None:
            self.threshold_reduction = [1,1]
        else:
            self.threshold_reduction = threshold_reduction
        self.inflation_factors = inflation_factors

    def set_epsilon(self, new_epsilon):
        self.epsilon = new_epsilon
        self.rho = 1 / (1 + np.exp(new_epsilon))

    def construct_response(self, m_i, mal_users):
        rho = self.rho
        C = self.inflation_factors
        input_graph = self.input_graph.get_adj_matrix()[m_i, :].copy()
        input_pois = input_graph.copy()
        input_pois[m_i] = 1
        input_pois[mal_users] = 1
        hon_0s = np.where(input_pois == 0)[0]
        connect_0s = min( int(C[0] * hon_0s.size), hon_0s.size)
        input_pois[hon_0s[:connect_0s]] = 1
        input_pois[m_i] = 0
        return input_pois

    def run_attack(self, attack, resp):
        mal_targets, hon_targets, mal_users = attack
        for mal_user in mal_users:
            resp[mal_user, hon_targets] = 0
            resp[mal_user, mal_targets] = 1
        
        for mal_user in mal_targets:
            cheat_resp = self.construct_response(mal_user, mal_users)
            resp[mal_user, :] = cheat_resp

        return resp

    def get_thresh(self):
        T = self.threshold_reduction
        i1 = self.max_num_mal * (1-2*self.rho) + T[1] * np.sqrt(8 * \
                                            max(self.n * self.rho,
                                                self.max_num_mal) *
                                              np.log(8 * self.n / self.delta))
        return T[0] * i1

    def get_degree_estimates(self):
        mat = self.input_graph.get_adj_matrix().copy()

        #Poison the input
        for a in self.attacks:
            mat = self.run_attack(a, mat)
        g_new = Graph(mat).get_noisy_graph(self.rho)
        noisy_resp = g_new.get_adj_matrix()

        thresh = self.get_thresh()
        self.set_epsilon(self.epsilon)
        deg_vec = np.zeros(self.n)
        for i in range(0, self.n):
            my_responses = noisy_resp[i, :]
            other_responses = noisy_resp[:, i]
            r_11 = (my_responses * other_responses).sum()
            r_01 = ((1-my_responses) * other_responses).sum()
            if (np.abs(r_01 - (self.n - 1) * self.rho *
                (1-self.rho))  >= thresh):
                deg_vec[i] = np.nan
                #print('rejected 1')
                continue
            rr_deg_est = 1.0 / (1-2 * self.rho) * \
                (r_11 - (self.n - 1) * self.rho**2)
            deg_vec[i] = rr_deg_est
        #pdb.set_trace()
        return deg_vec

class ResponseRRNaiveManipulation(ResponseRRManipulation):
    def __init__(self, input_graph, n_mal, epsilon, delta, inflation_factors, 
                 threshold_reduction = None):
        super(ResponseRRNaiveManipulation, self).__init__(
                input_graph, n_mal, epsilon, delta, inflation_factors, 
                threshold_reduction)

    def get_degree_estimates(self):
        mat = self.input_graph.get_adj_matrix().copy()

        hon_targets = set()
        for a in self.attacks:
            hon_targets = hon_targets.union(set(a[1]))

        g_new = Graph(mat).get_noisy_graph(self.rho)
        noisy_resp = g_new.get_adj_matrix()

        #Poison the responses
        for a in self.attacks:
            noisy_resp = self.run_attack(a, noisy_resp)
        thresh = self.get_thresh()
        self.set_epsilon(self.epsilon)
        deg_vec = np.zeros(self.n)
        for i in range(0, self.n):
            if i in hon_targets:
                other_responses = noisy_resp[:, i]
                r_1 = other_responses.sum()
            else:
                my_responses = noisy_resp[i, :]
                r_1 = my_responses.sum()
            deg_vec[i] = (r_1 - self.rho * self.n) / (1-2*self.rho)

        #pdb.set_trace()
        return deg_vec

class InputRRNaiveManipulation(InputRRManipulation):
    def __init__(self, input_graph, n_mal, epsilon, delta, inflation_factors, 
                 threshold_reduction = None):
        super(InputRRNaiveManipulation, self).__init__(
                input_graph, n_mal, epsilon, delta, inflation_factors,
                threshold_reduction)

    def get_degree_estimates(self):
        mat = self.input_graph.get_adj_matrix().copy()
				
        hon_targets = set()
        for a in self.attacks:
            hon_targets = hon_targets.union(set(a[1]))

        #Poison the input
        for a in self.attacks:
            mat = self.run_attack(a, mat)
        g_new = Graph(mat).get_noisy_graph(self.rho)
        noisy_resp = g_new.get_adj_matrix()

        thresh = self.get_thresh()
        self.set_epsilon(self.epsilon)
        deg_vec = np.zeros(self.n)
        for i in range(0, self.n):
            if i in hon_targets:
                other_responses = noisy_resp[:, i]
                r_1 = other_responses.sum()
            else:
                my_responses = noisy_resp[i, :]
                r_1 = my_responses.sum()
            deg_vec[i] = (r_1 - self.rho * self.n) / (1-2*self.rho)

        #pdb.set_trace()
        return deg_vec


def load_results_inc(arg):
    name, graph, n_mal, eps, delta, tr, seed, rlmi, rrmi, ilmi, irmi = arg
    np.random.seed(seed)
    print(seed)
    if name == 'RR+Laplace, Response':
        cls = ResponseLaplaceManipulationIncrease(graph.get_size(),
                                                     graph, n_mal, 0.0,
                                                     delta, 0.1, rlmi[0],
                                                     rlmi[1])
    elif name == 'RRCheck, Response':
        cls = ResponseRRManipulationIncrease(graph.get_size(), graph,
                                                n_mal, 0.0, delta, rrmi)
    elif name == 'RRNaive, Response':
        cls = ResponseNaiveRRManipulationIncrease(graph.get_size(), graph,
                                                n_mal, 0.0, delta)
    elif name == 'RR+Laplace, Input':
        cls = InputLaplaceManipulationIncrease(graph.get_size(), graph,
                                                  n_mal, 0.0, delta, 0.1,
                                                  ilmi[0], ilmi[1])
    elif name == 'RRCheck, Input':
        cls = InputRRManipulationIncrease(graph.get_size(), graph, n_mal,
                                             0.0, delta, irmi)
    elif name =='RRNaive, Input':
        cls = InputNaiveRRManipulationIncrease(graph.get_size(), graph, n_mal,
                                             0.0, delta)
    else:
        raise Exception('name %s not understood' % name)
    
    cls.choose_mal_and_target_inc()
    cls.set_epsilon(eps)
    cls.set_thresh_reduce(tr)
    hr, mr, ar = cls.get_results(verbose=False)
    return {'Algorithm': name, 
            'Num. Malicious': n_mal, 
            'Epsilon': eps, 
            'Delta': delta, 
            'Threshold Reduction': tr, 
            'Honest Num. Disq': hr[0],
            'Honest Avg Error': hr[1],
            'Honest Max Error': hr[2],
            'Malicious Num. Disq': mr[0],
            'Malicious Avg Error': mr[1],
            'Malicious Max Error': mr[2],
            'Target Disq': ar[0],
            'Target Error': ar[1]}
    
def load_results_dec(arg):
    name, graph, n_mal, eps, delta, tr, seed = arg
    np.random.seed(seed)
    print(seed)
    if name == 'RR+Laplace, Response':
        cls = ResponseLaplaceManipulationDecrease(graph.get_size(),
                                                     graph, n_mal, 0.0,
                                                     delta, 0.1)
    elif name == 'RRCheck, Response':
        cls = ResponseRRManipulationDecrease(graph.get_size(), graph,
                                                n_mal, 0.0, delta)
    elif name == 'RRNaive, Response':
        cls = ResponseNaiveRRManipulationDecrease(graph.get_size(), graph,
                                                n_mal, 0.0, delta)
    elif name == 'RR+Laplace, Input':
        cls = InputLaplaceManipulationDecrease(graph.get_size(), graph,
                                                  n_mal, 0.0, delta, 0.1)
    elif name == 'RRCheck, Input':
        cls = InputRRManipulationDecrease(graph.get_size(), graph, n_mal,
                                             0.0, delta)
    elif name =='RRNaive, Input':
        cls = InputNaiveRRManipulationDecrease(graph.get_size(), graph, n_mal,
                                             0.0, delta)
    else:
        raise Exception('name %s not understood' % name)
    
    cls.choose_mal_and_target_dec()
    cls.set_epsilon(eps)
    cls.set_thresh_reduce(tr)

    hr, mr, ar = cls.get_results(verbose=False)
    return {'Algorithm': name, 
            'Num. Malicious': n_mal, 
            'Epsilon': eps, 
            'Delta': delta, 
            'Threshold Reduction': tr, 
            'Honest Num. Disq': hr[0],
            'Honest Avg Error': hr[1],
            'Honest Max Error': hr[2],
            'Malicious Num. Disq': mr[0],
            'Malicious Avg Error': mr[1],
            'Malicious Max Error': mr[2],
            'Target Disq': ar[0],
            'Target Error': ar[1]}

def get_jobs(graph, n_mal, epsilon_list, delta, thresh,
                                     reps, rlmi, rrmi, ilmi, irmi):

    names = ['RR+Laplace, Response', 
             'RRCheck, Response',
             'RRNaive, Response',
             'RR+Laplace, Input',
             'RRCheck, Input',
             'RRNaive, Input']

    jobs = []
    seed = 10
    for name in names:
        for eps in epsilon_list:
            for tr in thresh:
                for i in range(0, reps):
                    seed += 11
                    if rlmi is not None:
                        jobs.append((name, graph, n_mal, eps, delta, tr, seed,
                                     rlmi, rrmi, ilmi, irmi))
                    else:
                        jobs.append((name, graph, n_mal, eps, delta, tr, seed))
    return jobs
         
from multiprocessing import Pool
if __name__ == '__main__':
    def run_test(dump_name):
        g = GNPGraph(100, 0.5)
        reps = 2
        thresh = [0.1, 0.2, 0.4, 0.6, 1.0]
        epsilons = [0.1,0.3,0.5,1,1.5,2,2.5,3.0]
        input_consts = ((0.1, 1.0), 0.1)
        response_consts = ((1.1, 1.1), 0.1)
        delta= 1e-5

        jobs1 = get_jobs(g, 10, epsilons, delta, thresh, reps, response_consts,
                         response_consts[0], input_consts, input_consts[0])
        jobs2 = get_jobs(g, 10, epsilons, delta, thresh, reps, None, None, None,
                         None)
        with Pool(12) as p:
            l1 = p.map(load_results_inc, jobs1)
            l2 = p.map(load_results_dec, jobs2)
            ans = l1 + l2
        """
        ans = []
        for i in jobs1:
            ans.append(load_results_inc(i))
        for i in jobs2:
            ans.append(load_results_dec(i))
        """
        pickle.dump(ans, open(dump_name, 'wb'))
    run_test('data/gnm_test.pkl')

    g = GNPGraph(4000, 0.5)
    fb = AdjacencyListGraph(4039, 'graphs/facebook_combined.txt')
    print("GNM No edges: %d" % g.get_adj_matrix().sum())
    print("GNM Inflation Asympt: %0.2f" % np.quantile(g.get_adj_matrix().sum(axis=1), 0.95))
    print("GNM Deflation Asympt: %0.2f" % (np.quantile(g.get_adj_matrix().sum(axis=1), 0.95) -
          np.quantile(g.get_adj_matrix().sum(axis=1), 0.80)))

    print("fb Inflation Asympt: %0.2f" % np.quantile(fb.get_adj_matrix().sum(axis=1), 0.95))
    print("fb Deflation Asympt: %0.2f" % (np.quantile(fb.get_adj_matrix().sum(axis=1), 0.95) -
          np.quantile(fb.get_adj_matrix().sum(axis=1), 0.80)))

    hi_mal = 1500
    low_mal = 40

    def run_real(g, dump_name, n_mal):

        reps = 50
        thresh = [0.05, 0.1, 0.2, 0.4, 0.6, 1.0]
        #epsilons = [0.1,0.3,0.5,1,1.5,2,2.5,3.0]
        epsilons = [0.3, 3.0]
        input_consts = ((0.1, 1.0), 0.1)
        response_consts = ((1.1, 1.1), 0.1)
        delta = 1e-5
        
        jobs1 = get_jobs(g, n_mal, epsilons, delta, thresh, reps, response_consts,
                         response_consts[0], input_consts, input_consts[0])
        jobs2 = get_jobs(g, n_mal, epsilons, delta, thresh, reps, None, None, None,
                         None)
        with Pool(12) as p:
            print("starting 1")
            l1 = p.map(load_results_inc, jobs1)
            print("starting 2")
            l2 = p.map(load_results_dec, jobs2)
        footer_inc = '_%d_inc.pkl' % n_mal
        footer_dec = '_%d_dec.pkl' % n_mal
        
        pickle.dump(l1, open(dump_name + footer_inc, 'wb'))
        pickle.dump(l2, open(dump_name + footer_dec, 'wb'))

    run_real(g, 'data_new/gnm', low_mal)
    run_real(g, 'data_new/gnm', hi_mal)
    #run_real(fb, 'data_new/fb', low_mal)
    #run_real(fb, 'data_new/fb', hi_mal)
    """
    epsilon_big = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    ExperimentTester.run_experiment_increase(g, 40, epsilon_big, 1e-6, reps,
                                              "data/gnm_inc_vary_epsilon.pkl",
                                              response_consts, response_consts[0],
                                              input_consts, input_consts[0])

    ExperimentTester.run_experiment_decrease(fb, 40, epsilon_big, 1e-6, reps,
                                              "data/fb_dec_vary_epsilon.pkl")

    """
