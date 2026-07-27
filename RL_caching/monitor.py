import math
import matplotlib.pyplot as plt
import numpy as np
import random

class Monitor:
    
    def __init__(self, num_servers, total_files, seed=None):
        self.num_servers = num_servers
        self.total_files = total_files
        self.file_to_server = {}
        top_10_percent = int(total_files * 0.10)

        rng = random.Random(seed) if seed is not None else random
        
        # Simula o desbalanceamento inicial alocando os ficheiros mais populares no Servidor 0
        for i in range(total_files):
            if i < top_10_percent:
                self.file_to_server[i] = 0 
            else:
                self.file_to_server[i] = rng.randint(0, num_servers - 1)
                
    def get_server(self, file_id):
        """ Aqui a gente pega o servidor que tem o arquivo requisitado """
        return self.file_to_server[file_id]
    
    def jains_fairness_index(self, loads):
        loads = np.asarray(loads, dtype=float)
        sum_loads = np.sum(loads)
        sum_sq_loads = np.sum(loads**2)
        
        # Previne divisão por zero caso o omega esteja vazio no início
        if sum_sq_loads == 0:
            return 1.0
            
        return (sum_loads**2) / (self.num_servers * sum_sq_loads)


    def plot_hit_rate(self, cache, req, popularities, c=None, algorithm=None, plot=False):
        hits_count = 0
        hit_curve = [] if plot else None

        omega = np.zeros(self.num_servers)

        for i, f in enumerate(req, 1):
            if f in cache.state:
                is_hit = 1 
            else:
                is_hit = 0
                server_id = self.get_server(f)
                omega[server_id] += 1
            hits_count += is_hit

            try:
                cache.policy(f, omega)
            except TypeError:
                cache.policy(f)

            if plot:
                hit_curve.append(hits_count / i)

        hit_rate = hits_count / len(req) if len(req) else 0.0
        jfi = self.jains_fairness_index(omega)
        angle_degrees = float(np.degrees(np.arccos(np.sqrt(min(max(jfi, 0.0), 1.0)))))

        result = {
            "algorithm": algorithm,
            "hit_rate": hit_rate,
            "jfi": jfi,
            "angle_degrees": angle_degrees,
            "omega": omega.tolist(),
        }
        if plot:
            result["hit_curve"] = hit_curve
        return result

    def plot_hit_rate(self, cache, req, popularities, c, algorithm):

        result = self.run_metrics(cache, req, popularities, c, algorithm, plot=True)

        plt.axhline(y = sum(popularities[:c]), color = 'C0', linestyle = '--', label='Optimal')
        plt.plot(result["hit_curve"])
        plt.show()

        print("%s's Hit Rate: %.2f%%" % (algorithm, result["hit_rate"] * 100))
        print("Omega:", np.array(result["omega"]))
        print("Jain's Fairness Index: %.4f" % result["jfi"])
        print("Angle to Perfect Fairness Line: %.2f degrees" % result["angle_degrees"])
        print("-" * 50)
        return result

    def run_metrics(self, cache, req, popularities=None, c=None, algorithm=None, plot=False):
        hits_count = 0
        hit_curve = [] if plot else None
        omega = np.zeros(self.num_servers)

        for i, f in enumerate(req, 1):
            if f in cache.state:
                is_hit = 1
            else:
                is_hit = 0
                server_id = self.get_server(f)
                omega[server_id] += 1
            hits_count += is_hit

            try:
                cache.policy(f, omega)
            except TypeError:
                cache.policy(f)

            if plot:
                hit_curve.append(hits_count / i)

        hit_rate = hits_count / len(req) if len(req) else 0.0
        jfi = self.jains_fairness_index(omega)
        angle_degrees = float(np.degrees(np.arccos(np.sqrt(min(max(jfi, 0.0), 1.0)))))

        result = {
            "algorithm": algorithm,
            "hit_rate": hit_rate,
            "jfi": jfi,
            "angle_degrees": angle_degrees,
            "omega": omega.tolist(),
        }
        if plot:
            result["hit_curve"] = hit_curve
        return result