import math
import matplotlib.pyplot as plt
import numpy as np
import random

class Monitor:
    
    def __init__(self, num_servers, total_files, seed=None, skew_fraction=0.0):
        """
        skew_fraction: fracao do catalogo (por INDICE de arquivo, os IDs
            0..int(total_files*skew_fraction)-1) que fica fixa, sempre, no
            servidor 0 -- independente de num_servers. E um bloco de tamanho
            fixo, entao conforme num_servers cresce ele fica proporcionalmente
            mais concentrado que os demais servidores (que dividem o resto do
            catalogo entre si). Isso NAO depende de popularidade de fato: se
            as popularidades vierem de zipf(..., seed=...), elas ja estao
            embaralhadas em relacao ao indice do arquivo, entao esse bloco e
            so um subconjunto fixo de arquivos, nao necessariamente os mais
            acessados.

            skew_fraction=0.0 (default): distribuicao totalmente aleatoria e
                uniforme entre os servidores, sem hotspot estrutural. E o que
                a maioria dos experimentos de escalabilidade deveria usar.
            skew_fraction=0.10: reproduz o comportamento historico deste
                projeto (10% do catalogo sempre no servidor 0), util para
                comparar "com hotspot" vs "sem hotspot" lado a lado.
        """
        self.num_servers = num_servers
        self.total_files = total_files
        self.file_to_server = {}
        skew_count = int(total_files * skew_fraction)

        rng = random.Random(seed) if seed is not None else random

        for i in range(total_files):
            if i < skew_count:
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


    def plot_hit_rate(self, cache, req, popularities, c, algorithm, workload=None):
        result = self.run_metrics(cache, req, popularities, c, algorithm, plot=True, workload=workload)

        plt.axhline(y = sum(popularities[:c]), color = 'C0', linestyle = '--', label='Optimal')
        plt.plot(result["hit_curve"])
        plt.show()

        print("%s's Hit Rate: %.2f%%" % (algorithm, result["hit_rate"] * 100))
        print("Omega:", np.array(result["omega"]))
        print("Jain's Fairness Index: %.4f" % result["jfi"])
        print("Angle to Perfect Fairness Line: %.2f degrees" % result["angle_degrees"])
        print("-" * 50)
        return result

    def run_metrics(self, cache, req, popularities=None, c=None, algorithm=None, plot=False, workload=None):
        """
        workload: array indexado por file_id com o "custo" de uma requisicao
            perdida (cache miss) para aquele arquivo. Se None (default), todo
            miss conta como 1 unidade de carga, independente do arquivo --
            era o comportamento (implicito) anterior. Se fornecido, cada
            miss ao arquivo f soma workload[f] unidades de carga ao servidor
            que o hospeda, em vez de sempre 1.
        """
        hits_count = 0
        hit_curve = [] if plot else None
        omega = np.zeros(self.num_servers)

        for i, f in enumerate(req, 1):
            if f in cache.state:
                is_hit = 1
            else:
                is_hit = 0
                server_id = self.get_server(f)
                omega[server_id] += 1.0 if workload is None else float(workload[f])
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