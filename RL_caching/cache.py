import math
import random
import numpy as np
from collections import OrderedDict
"""
- LRU, QLRU e Optimal_QLRU guardam self.state como um OrderedDict, em vez de uma lista.
- A complexidade caiu de O(C) para O(1)
"""
class LRU:
    def __init__(self, _state, _c):
        self.state = OrderedDict((f, True) for f in _state)
        self.c = _c

    def insert(self, f, pos):
        self.state[f] = True
        
    def delete(self, pos):
        self.state.popitem(last=False)
    
    def move(self, f, pos):
        self.state.move_to_end(f)

    def policy(self, f):
        if f in self.state:
            self.move(f,-1)
        else:
            if len(self.state) == self.c:
                self.delete(0)
            self.insert(f,-1)

class QLRU:
    def __init__(self, _state, _c, _q):
        self.state = OrderedDict((f, True) for f in _state)
        self.c = _c
        self.q = _q

    def insert(self, f, pos):
        self.state[f] = True
        
    def delete(self, pos):
        self.state.popitem(last=False)
    
    def move(self, f, pos):
        self.state.move_to_end(f)

    def policy(self, f):
        if f in self.state:
            self.move(f,-1)
        else:
            if random.uniform(0,1) < self.q:   
                if len(self.state) == self.c:
                    self.delete(0)
                self.insert(f,-1)


class LFU:
    def __init__(self, _state, _c, _catalog_size):
        self.state = _state
        self.c = _c
        self.q = _catalog_size
        self.counter = {i: 1 for i in self.state}

    def insert(self, f, pos):
        if pos == -1:
            self.state.append(f)
        else:
            self.state.insert(pos, f)
        
    def delete(self, pos):
        self.state.pop(pos)
    
    def move(self, f, pos):
        self.state.remove(f)
        self.state.insert(pos, f)

    def policy(self, f):
        if f in self.state:
            self.counter[f] += 1
        else:
            if len(self.state) == self.c:
                f_out = min(self.counter, key=self.counter.get)
                del self.counter[f_out]
                self.delete(self.state.index(f_out))
            self.insert(f, -1)
            self.counter[f] = 1

class IRM:
    """ Aqui pegamos os ultimos objetos c do cache e invertemos para ter do mais popular para o menos """
    def __init__(self, _c, _popularities):
        self.c = _c
        self.state = list(np.argsort(_popularities)[-self.c:][::-1])

    def policy(self, f):
        """ O estado do cache nao muda """
        pass

class FairStatic:
    """ Nessa classe, primeiro calcula a carga inicial de cada servidor baseado nas popularidades dos arquivos e em qual servidor cada arquivo está. Depois, seleciona os c arquivos mais populares de forma a balancear a carga entre os servidores. """
    def __init__(self, _c, _popularities, _num_servers, _file_to_server): 
        self.c = _c
        self.state = []

        loads = np.zeros(_num_servers)
        for i, p in enumerate(_popularities):
            loads[_file_to_server[i]] += p

        self.initial_loads = loads.copy()
        
        available_files = list(np.argsort(_popularities)[::-1])
        """ Encontra o servidor mais carregado e adiciona o arquivo mais popular que está nesse servidor ao cache, repetindo até que o cache esteja cheio. """
        for _ in range(self.c):
            heaviest_server = np.argmax(loads)
            
            for f in available_files:
                if _file_to_server[f] == heaviest_server and f not in self.state:
                    self.state.append(f)
                    loads[heaviest_server] -= _popularities[f]
                    break
    def policy(self, f):
        pass



class Optimal_QLRU:
    def __init__(self, _state, _c, _beta, _sizes, file_to_server, rng=None, track_stats=False):
        """
        rng: instancia de random.Random usada na decisao probabilistica de
            admissao (random.uniform(0,1) < q_f). Se None, cria uma
            instancia SEM seed fixa (nao reprodutivel). O chamador deveria
            sempre passar um random.Random(seed) explicito para
            reprodutibilidade -- o modulo `random` global nao e sincronizado
            entre processos sob multiprocessing.

        track_stats: se True, guarda em self.q_history o valor de q_f
            calculado em CADA decisao de admissao (cache miss), em
            self.delta_jfi_history o delta_jfi correspondente, em
            self.roll_history o numero sorteado (self.rng.uniform(0,1))
            usado na decisao, e em self.decision_history se foi admitido
            (True) ou nao (False). Usado para estatisticas de como "q" esta
            sendo atribuido (media, mediana, desvio padrao) e para
            depuracao/analise/visualizacao. Custa memoria O(numero de
            misses) quando ligado -- deixe False em rodadas grandes que nao
            precisem dessa analise.
        """
        self.state = OrderedDict((f, True) for f in _state)
        self.c = _c           
        self.beta = _beta    
        self.sizes = _sizes    
        self.file_to_server = file_to_server 
        self.rng = rng if rng is not None else random.Random()
        self.track_stats = track_stats
        self.q_history = [] if track_stats else None
        self.delta_jfi_history = [] if track_stats else None
        self.roll_history = [] if track_stats else None
        self.decision_history = [] if track_stats else None

        self.current_occupancy = sum(self.sizes[f] for f in self.state)

    def insert(self, f, pos):
        self.state[f] = True
        self.current_occupancy += self.sizes[f]
        
    def delete(self, pos):  
        f_removido = self.state.popitem(last=False)
        self.current_occupancy -= self.sizes[f_removido[0]]
    
    def move(self, f, pos):
        self.state.move_to_end(f)

    def _calculate_jfi(self, loads):
        """ Função auxiliar para calcular o Jain's Fairness Index """
        loads = np.asarray(loads, dtype=np.float64)
        sum_loads = sum(loads)
        if sum_loads == 0:
            return 1.0 # Sistema perfeitamente justo se não houver carga em nenhum servidor
        
        sum_sq_loads = np.square(loads).sum()
        num_servers = loads.shape[0]
        
        return (sum_loads**2) / (num_servers * sum_sq_loads)


    def policy(self, f, current_omega, fix_double_count=True):
        """
        fix_double_count: por padrao True (comportamento correto). Existe
        so para permitir comparar contra o comportamento antigo/com bug --
        current_omega, quando chega aqui, JA inclui a carga da requisicao
        atual (Monitor.run_metrics incrementa omega[server_id] ANTES de
        chamar policy()). A versao antiga (fix_double_count=False) somava
        +1 de novo no cenario "sem cache" e nao descontava nada no cenario
        "com cache", contando a mesma requisicao de forma inconsistente
        entre os dois cenarios simulados. Mantenha True a menos que esteja
        reproduzindo/comparando com resultados antigos de proposito.
        """
        if f in self.state:
            self.move(f, -1)
        else:
            s_f = self.sizes[f]
            server_id = self.file_to_server[f]

            if fix_double_count:
                # current_omega JA inclui a carga desta requisicao (o
                # Monitor incrementa omega[server_id] ANTES de chamar
                # policy()) -- entao "sem cache" e o proprio current_omega
                # sem modificacao, e "com cache" e current_omega MENOS 1
                # (desfaz a carga que nao teria acontecido se fosse hit).
                loads_sem_cache = np.array(current_omega, dtype=np.float64)
                jfi_sem = self._calculate_jfi(loads_sem_cache)

                loads_com_cache = np.array(current_omega, dtype=np.float64)
                loads_com_cache[server_id] -= 1
                jfi_com = self._calculate_jfi(loads_com_cache)
            else:
                # 1. Simula o JFI se o ficheiro NÃO for para o cache (Cache Miss total)
                # A carga do servidor de origem aumenta com este pedido.
                loads_sem_cache = np.array(current_omega, dtype=np.float64) # Faz uma cópia da carga atual
                loads_sem_cache[server_id] += 1       # Assumimos 1 requisição (ou += s_f se a carga for medida em tamanho)
                jfi_sem = self._calculate_jfi(loads_sem_cache)

                # 2. Simula o JFI se o ficheiro FOR para o cache
                # A carga do servidor fica como está (o cache absorve o impacto)
                loads_com_cache = list(current_omega)
                jfi_com = self._calculate_jfi(loads_com_cache)

            # 3. Calcula o Delta JFI (o impacto matemático deste ficheiro na justiça global)
            delta_jfi = jfi_com - jfi_sem
            
            # 4. Transforma o Delta numa probabilidade q_f
            if delta_jfi > 0:
                # O ficheiro AJUDA o balanceamento! 
                # Como o delta_jfi é um número decimal muito pequeno (ex: 0.005), 
                # multiplicamos por uma constante (ex: 100) para que a fórmula exponencial ganhe escala.
                # A variável beta continua a controlar a sensibilidade da IA.
                q_f = 1.0 - math.exp(-self.beta * (delta_jfi * 100))
            else:
                # O ficheiro PREJUDICA ou não tem impacto no balanceamento.
                # Não o queremos no cache.
                q_f = 0.0

            if self.track_stats:
                self.q_history.append(q_f)
                self.delta_jfi_history.append(delta_jfi)

            # 5. Roda a roleta para decidir a admissão no cache
            roll = self.rng.uniform(0, 1)
            admitiu = roll < q_f
            if self.track_stats:
                self.roll_history.append(roll)
                self.decision_history.append(admitiu)
            if admitiu:
                if s_f <= self.c:
                    while s_f > (self.c - self.current_occupancy):
                        self.delete(0)
                    self.insert(f, -1)


class Optimal_QLRU_2:
    """
    Variante que implementa a formula classica de q-LRU, substituindo t_f
    (tempo de recuperacao) por delta_jfi (impacto marginal desta admissao no
    Jain's Fairness Index global) -- exatamente a substituicao pedida:

        q_f = e^(-beta * s_f / t_f)   [formula original, t_f = retrieval time]
        q_f = e^(-beta * s_f / (delta_jfi_scale * delta_jfi))   [substituicao]

    delta_jfi e calculado do mesmo jeito que em Optimal_QLRU (simula o JFI
    global com e sem cachear este arquivo).

    delta_jfi_scale existe porque delta_jfi tende a ser MUITO menor que um
    t_f tipico (a contribuicao marginal de UM arquivo no JFI de dezenas/
    centenas de servidores e um numero pequeno, tipo 1e-4 ou menor) -- sem
    escalar, a formula literal da q_f praticamente sempre ~0 e o cache nunca
    admite nada. O valor default (23000) foi calibrado empiricamente neste
    projeto (ver conversa) para dar um comportamento nao-degenerado; ajuste
    conforme o cenario (catalog_size, num_servers, capacity) se os
    resultados saírem sempre proximos de 0 ou sempre proximos de 1.

    Diferente da formula original (onde t_f nunca e zero/negativo), delta_jfi
    PODE ser zero ou negativo -- nesses casos usamos q_f=0 diretamente (a
    formula literal nao tem essa guarda porque nao precisa dela).
    """
    def __init__(self, _state, _c, _beta, _sizes, file_to_server, rng=None, track_stats=False,
                 delta_jfi_scale=23000.0):
        self.state = OrderedDict((f, True) for f in _state)
        self.c = _c
        self.beta = _beta
        self.sizes = _sizes
        self.file_to_server = file_to_server
        self.rng = rng if rng is not None else random.Random()
        self.track_stats = track_stats
        self.q_history = [] if track_stats else None
        self.delta_jfi_history = [] if track_stats else None
        self.delta_jfi_scale = delta_jfi_scale

        self.current_occupancy = sum(self.sizes[f] for f in self.state)

    def insert(self, f, pos):
        self.state[f] = True
        self.current_occupancy += self.sizes[f]

    def delete(self, pos):
        f_removido = self.state.popitem(last=False)
        self.current_occupancy -= self.sizes[f_removido[0]]

    def move(self, f, pos):
        self.state.move_to_end(f)

    def _calculate_jfi(self, loads):
        loads = np.asarray(loads, dtype=np.float64)
        sum_loads = sum(loads)
        if sum_loads == 0:
            return 1.0
        sum_sq_loads = np.square(loads).sum()
        num_servers = loads.shape[0]
        return (sum_loads**2) / (num_servers * sum_sq_loads)

    def policy(self, f, current_omega):
        if f in self.state:
            self.move(f, -1)
        else:
            s_f = self.sizes[f]
            server_id = self.file_to_server[f]

            # current_omega ja inclui a carga desta requisicao (Monitor
            # incrementa antes de chamar policy()) -- "sem cache" e o
            # proprio current_omega, "com cache" desconta essa carga
            loads_sem_cache = np.array(current_omega, dtype=np.float64)
            jfi_sem = self._calculate_jfi(loads_sem_cache)

            loads_com_cache = np.array(current_omega, dtype=np.float64)
            loads_com_cache[server_id] -= 1
            jfi_com = self._calculate_jfi(loads_com_cache)

            delta_jfi = jfi_com - jfi_sem

            if delta_jfi > 0:
                # formula classica de q-LRU: q_f = e^(-beta * s_f / t_f),
                # com t_f substituido por (delta_jfi_scale * delta_jfi)
                q_f = math.exp(-self.beta * s_f / (self.delta_jfi_scale * delta_jfi))
            else:
                q_f = 0.0

            if self.track_stats:
                self.q_history.append(q_f)
                self.delta_jfi_history.append(delta_jfi)

            if self.rng.uniform(0, 1) < q_f:
                if s_f <= self.c:
                    while s_f > (self.c - self.current_occupancy):
                        self.delete(0)
                    self.insert(f, -1)