import matplotlib.pyplot as plt
import numpy as np

class Monitor:
    
    def __init__(self, num_servers, total_files):
        self.num_servers = num_servers
        self.total_files = total_files

    def get_server(self, file_id):
        server_size = self.total_files // self.num_servers
        server_id = file_id // server_size
        return min(server_id, self.num_servers - 1)
    
    def jains_fairness_index(self, loads):
        sum_loads = np.sum(loads)
        sum_sq_loads = np.sum(loads**2)

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
            cache.policy(f)
            hit_aux.append(hits_count / i)

        plt.axhline(y = sum(popularities[:c]), color = 'C0', linestyle = '--', label='Optimal')
        plt.plot(hit_aux)
        plt.show()

        hit_rate = hits_count / len(req)
        print("%s's Hit Rate: %.2f%%" % (algorithm, hit_rate * 100))

        hit_rate = hits_count / len(req)
        jfi = self.jains_fairness_index(omega)
        print("Omega:", omega)
        print("Jain's Fairness Index: %.4f" % jfi)

        angle_radians = np.arctan2(omega[1], omega[0])
        angle_degrees = np.degrees(angle_radians)

        print("Angle of Vector Ω: %.2f degrees" % angle_degrees)
