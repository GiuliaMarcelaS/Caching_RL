import math
import matplotlib.pyplot as plt
import numpy as np
import random

class Monitor:
    
    def __init__(self, num_servers, total_files):
        self.num_servers = num_servers
        self.total_files = total_files
        self.file_to_server = {}
        top_10_percent = int(total_files * 0.10)
        
        # Simula o desbalanceamento inicial alocando os ficheiros mais populares no Servidor 0
        for i in range(total_files):
            if i < top_10_percent:
                self.file_to_server[i] = 0 
            else:
                self.file_to_server[i] = random.randint(0, num_servers - 1)
                
    def get_server(self, file_id):
        """ Aqui a gente pega o servidor que tem o arquivo requisitado """
        return self.file_to_server[file_id]
    
    def jains_fairness_index(self, loads):
        sum_loads = np.sum(loads)
        sum_sq_loads = np.sum(loads**2)
        
        # Previne divisão por zero caso o omega esteja vazio no início
        if sum_sq_loads == 0:
            return 1.0
            
        return (sum_loads**2) / (self.num_servers * sum_sq_loads)


    def plot_hit_rate(self, cache, req, popularities, c, algorithm):
        hits_count = 0
        hit_aux = []

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
            hit_aux.append(hits_count / i)

        plt.axhline(y = sum(popularities[:c]), color = 'C0', linestyle = '--', label='Optimal')
        plt.plot(hit_aux)
        plt.show()

        hit_rate = hits_count / len(req)
        print("%s's Hit Rate: %.2f%%" % (algorithm, hit_rate * 100))

        jfi = self.jains_fairness_index(omega)
        print("Omega:", omega)
        print("Jain's Fairness Index: %.4f" % jfi)

        # --- A GRANDE ALTERAÇÃO MATEMÁTICA AQUI ---
        # Como cos^2(theta) = JFI, então o ângulo (theta) é o arco-cosseno da raiz quadrada do JFI.
        # Esta matemática funciona não importa quantos servidores (dimensões) você tenha!
        angle_radians = np.arccos(np.sqrt(jfi))
        angle_degrees = np.degrees(angle_radians)

        print("Angle to Perfect Fairness Line: %.2f degrees" % angle_degrees)
        print("-" * 50)