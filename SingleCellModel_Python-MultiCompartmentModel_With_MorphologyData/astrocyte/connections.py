"""
from peter
"""
from utils.writable import Writable

class Connections(Writable):
    
    _name = "Connections"
    
    def __init__(self, d):
        self._d = d
        self.get_comp_connections()
        
    def get_comp_connections(self):
        for i, comps in enumerate(self._d):
            if i == 0: # for the soma
                continue
            for connection in self._d[comps]._connections:
                self._d[comps]._comp_connections += [f"compartment_{connection}"] # adds connections
            if (i != 0) and (self._d[comps]._comp_lvl == 1): # if not soma and process_lvl is 1
                    self._d["compartment_0"]._comp_connections += [comps] # adds connections to the soma
                    self._d[comps]._comp_connections += ["compartment_0"] # adds soma connection to compartment
                    
    def get_name(self):
        return self._name
                    
                    
                
                