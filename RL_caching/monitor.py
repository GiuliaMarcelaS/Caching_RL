import math
import matplotlib.pyplot as plt
import numpy as np
import random

class Monitor:
    
    def __init__(self, num_servers, total_files, seed=None, skew_fraction=0.0,
                 skew_mode="index", popularities=None, exclude_top_k=0):
        """
        skew_fraction: fracao do catalogo que fica fixa, sempre, no servidor
            0 -- independente de num_servers. E um bloco de tamanho fixo,
            entao conforme num_servers cresce ele fica proporcionalmente
            mais concentrado que os demais servidores (que dividem o resto
            do catalogo entre si).

            skew_fraction=0.0 (default): distribuicao totalmente aleatoria e
                uniforme entre os servidores, sem hotspot estrutural.
            skew_fraction=0.10: reproduz o comportamento historico deste
                projeto (10% do catalogo sempre no servidor 0).

        skew_mode: como escolher QUAIS arquivos formam o bloco fixo.
            "index" (default): os primeiros skew_fraction*total_files
                INDICES de arquivo. Se as popularidades vierem de
                zipf(..., seed=...), elas ja estao embaralhadas em relacao
                ao indice, entao isso equivale, na pratica, a um
                subconjunto ALEATORIO do catalogo -- sem relacao deliberada
                com popularidade.
            "exclude_top_k": o bloco fixo do servidor 0 e formado pelos
                skew_fraction*total_files arquivos com MAIOR popularidade
                DENTRE OS QUE NAO ESTAO nos exclude_top_k mais populares do
                catalogo inteiro (requer `popularities`). Ou seja: os
                arquivos globalmente mais populares (ranks 1..exclude_top_k)
                NUNCA vao pro servidor 0, mas o bloco escolhido ainda e
                popular o bastante (por ser o "topo do resto") pra deixar o
                servidor 0 genuinamente sobrecarregado (mais carga agregada
                que a fracao justa 1/num_servers), mesmo excluindo a elite.

                Serve pra construir cenarios onde o cache OTIMO PARA HIT
                RATE (cachear os arquivos mais populares do catalogo, onde
                quer que estejam) da ZERO alivio ao servidor 0 -- ja que
                nenhum desses arquivos esta la -- enquanto o cache OTIMO
                PARA FAIRNESS precisa especificamente de conteudo do
                servidor 0 (que nao esta entre os mais populares) pra
                equilibrar a carga. E exatamente o cenario "o resultado
                otimo de fairness nao e colocar no cache os arquivos mais
                populares".
        exclude_top_k: usado so em skew_mode="exclude_top_k" -- quantos dos
            arquivos globalmente mais populares ficam de fora do bloco do
            servidor 0 (tipicamente = capacity do cache, pra modelar "o que
            um cache guloso por hit rate escolheria").
        """
        self.num_servers = num_servers
        self.total_files = total_files
        self.file_to_server = {}
        skew_count = int(total_files * skew_fraction)

        rng = random.Random(seed) if seed is not None else random

        if skew_mode == "index":
            skewed_files = set(range(skew_count))
        elif skew_mode == "exclude_top_k":
            if popularities is None:
                raise ValueError("skew_mode='exclude_top_k' requer popularities")
            popularities = np.asarray(popularities)
            order = np.argsort(-popularities)  # descendente: mais populares primeiro
            candidatos = order[exclude_top_k:]  # exclui os exclude_top_k mais populares
            skewed_files = set(candidatos[:skew_count].tolist())
        else:
            raise ValueError(f"skew_mode desconhecido: {skew_mode!r} (use 'index' ou 'exclude_top_k')")

        for i in range(total_files):
            if i in skewed_files:
                self.file_to_server[i] = 0
            elif skew_count > 0:
                # servidor 0 e reservado para o bloco marcado -- exclui-lo do
                # sorteio evita "diluir" o efeito com arquivos aleatorios
                # extras caindo la por sorte
                self.file_to_server[i] = rng.randint(1, num_servers - 1)
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

    def run_metrics(self, cache, req, popularities=None, c=None, algorithm=None, plot=False, workload=None,
                     curve_points=None):
        """
        workload: array indexado por file_id com o "custo" de uma requisicao
            perdida (cache miss) para aquele arquivo. Se None (default), todo
            miss conta como 1 unidade de carga, independente do arquivo --
            era o comportamento (implicito) anterior. Se fornecido, cada
            miss ao arquivo f soma workload[f] unidades de carga ao servidor
            que o hospeda, em vez de sempre 1.

        curve_points: se um inteiro for passado, registra a evolucao do JFI
            ao longo da requisicoes (amostrado em ~curve_points pontos
            igualmente espacados), retornado em result["jfi_curve"] como
            lista de (indice_da_requisicao, jfi_naquele_momento). Serve para
            analisar convergencia (JFI x tempo), do mesmo jeito que
            run_qlearning ja faz para o agente de Q-learning. None (default)
            = nao rastreia (economiza custo em rodadas grandes que nao
            precisem disso).
        """
        hits_count = 0
        hit_curve = [] if plot else None
        jfi_curve = [] if curve_points else None
        curve_every = max(1, len(req) // curve_points) if curve_points else None
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

            if curve_points and (i == 1 or i % curve_every == 0 or i == len(req)):
                jfi_curve.append((i, self.jains_fairness_index(omega)))

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
        if curve_points:
            result["jfi_curve"] = jfi_curve
        return result