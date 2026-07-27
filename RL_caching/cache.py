import math
import random
import numpy as np

class LRU:
    def __init__(self, _state, _c):
        self.state = _state
        self.c = _c

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
            self.move(f,-1)
        else:
            if len(self.state) == self.c:
                self.delete(0)
            self.insert(f,-1)

class QLRU:
    def __init__(self, _state, _c, _q):
        self.state = _state
        self.c = _c
        self.q = _q

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
    def __init__(self, _state, _c, _beta, _sizes, file_to_server):
        self.state = _state
        self.c = _c           
        self.beta = _beta    
        self.sizes = _sizes    
        self.file_to_server = file_to_server 
        

        self.current_occupancy = sum(self.sizes[f] for f in self.state)

    def insert(self, f, pos):
        if pos == -1:
            self.state.append(f)
        else:
            self.state.insert(pos, f)

        self.current_occupancy += self.sizes[f]
        
    def delete(self, pos):
        f_removido = self.state.pop(pos)

        self.current_occupancy -= self.sizes[f_removido]
    
    def move(self, f, pos):
        self.state.remove(f)
        self.state.insert(pos, f)

    def _calculate_jfi(self, loads):
        """ Função auxiliar para calcular o Jain's Fairness Index """
        sum_loads = sum(loads)
        if sum_loads == 0:
            return 1.0 # Sistema perfeitamente justo se não houver carga em nenhum servidor
        
        sum_sq_loads = sum(l**2 for l in loads)
        num_servers = len(loads)
        
        return (sum_loads**2) / (num_servers * sum_sq_loads)


    def policy(self, f, current_omega):
        if f in self.state:
            self.move(f, -1)
        else:
            s_f = self.sizes[f]
            server_id = self.file_to_server[f]
            
            # 1. Simula o JFI se o ficheiro NÃO for para o cache (Cache Miss total)
            # A carga do servidor de origem aumenta com este pedido.
            loads_sem_cache = list(current_omega) # Faz uma cópia da carga atual
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

            # 5. Roda a roleta para decidir a admissão no cache
            if random.uniform(0, 1) < q_f:   
                if s_f <= self.c:
                    while s_f > (self.c - self.current_occupancy):
                        self.delete(0)
                    self.insert(f, -1)