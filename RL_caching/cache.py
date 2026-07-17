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
    def __init__(self, _c, _popularities, _num_servers, _file_to_server, _q=0.01, _dynamic=True):
        self.c = _c
        self.num_servers = _num_servers
        self.file_to_server = _file_to_server
        self.dynamic = _dynamic
        self.base_q = _q
        self.state = []
 
        # Warm-start estatico: alocacao inicial que balanceia a carga 
        loads = np.zeros(_num_servers)
        for i, p in enumerate(_popularities):
            loads[_file_to_server[i]] += p
 
        self.initial_loads = loads.copy()
 
        available_files = list(np.argsort(_popularities)[::-1])
        # Encontra o servidor mais carregado e adiciona o arquivo mais popular que
        # esta nesse servidor ao cache, repetindo ate o cache encher.
        for _ in range(self.c):
            heaviest_server = np.argmax(loads)
 
            for f in available_files:
                if _file_to_server[f] == heaviest_server and f not in self.state:
                    self.state.append(f)
                    loads[heaviest_server] -= _popularities[f]
                    break
 
        # Comeca em 1 para evitar divisao por zero no primeiro acesso.
        self.loads = np.ones(_num_servers)
 
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
        # Se for a versao puramente estatica, o cache nunca muda (baseline).
        if not self.dynamic:
            return
 
        s = self.file_to_server[f]
 
        if f in self.state:
            # HIT: atualiza a posicao no estilo LRU (move para o fim da fila).
            self.move(f, -1)
        else:
            # MISS:  adiciona carga ao servidor s.
            self.loads[s] += 1
 
            # Admissao consciente da carga: servidor mais carregado -> maior prob.
            # q_f fica em (0, base_q]; vale base_q exatamente para o servidor de
            # carga maxima e cai proporcionalmente para os menos carregados.
            q_f = self.base_q * (self.loads[s] / self.loads.max())
 
            if random.uniform(0, 1) < q_f:
                if len(self.state) == self.c:
                    self.delete(0)      # remove o mais antigo
                self.insert(f, -1)



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


    def policy(self, f, current_omega):
        if f in self.state:
            self.move(f, -1)
        else:

            s_f = self.sizes[f]

            server_id = self.file_to_server[f]
            
            server_load = current_omega[server_id] + 1
            

            q_f = math.exp(-self.beta * (s_f / server_load))
            

            if random.uniform(0, 1) < q_f:   
                
              
                if s_f <= self.c:
                    

                    while s_f > (self.c - self.current_occupancy):
                        self.delete(0)
                    
                    self.insert(f, -1)