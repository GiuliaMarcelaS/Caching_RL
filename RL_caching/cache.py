import random

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