"""
from peter
"""
import numpy as np
from configparser import ConfigParser
import ast

from utils.writable import Writable


class Geometry(Writable):
    
    _name = "Geometry"
    
    """
    soma + multiple processes
    """
    # proc = [[1,1,0,0],[1,1,1,1],[0,1,1,0],[0,1,0,1]] # this will be in a config file
    # proc_2 = [[1,1],[1,1]] # this will be in a config file
    # proc_3 = [[1,1,0,1,1,0],[1,1,1,0,0,0],[0,1,1,0,0,0],[1,0,0,1,0,0],[1,0,0,0,1,1],[0,0,0,0,1,1]] # this will be in a config file
    # astrocyte_1 = [proc_3,0,proc,0,proc_2,0] # this will be in a config file
    
    """
    just the soma
    """
    #astrocyte_1 = [] # this will be in a config file
    
    """
    soma + 1 compartment
    """
    # astrocyte_1 = [[1]] # this will be in a config file
    
    """
    soma + multiple processes
    """
    # astrocyte_1 = [[[1,1],[1,1]], [1]] # this will be in a config file
    
    """
    gordleeva
    """
    # proc_1 = [[1,1,0,0,0,1,0,0],[1,1,1,0,1,0,0,0],[0,1,1,1,0,0,0,0],[0,0,1,1,0,0,0,0],[0,1,0,0,1,0,0,0],
    #           [1,0,0,0,0,1,1,1],[0,0,0,0,0,1,1,0],[0,0,0,0,0,1,0,1]]
    # proc_2 = [[1,1,0,0,0],[1,1,1,0,0],[0,1,1,1,0],[0,0,1,1,1],[0,0,0,1,1]]
    # proc_3 = [[1,1,0,0,0],[1,1,1,0,0],[0,1,1,1,0],[0,0,1,1,1],[0,0,0,1,1]]
    # proc_4 = [[1,1,0,0,0,0],[1,1,1,0,0,0],[0,1,1,1,0,0],[0,0,1,1,1,0],[0,0,0,1,1,1],[0,0,0,0,1,1]]
    # proc_5 = [[1,1,0,1,0,0,0,0],[1,1,1,0,0,0,0,0],[0,1,1,0,0,0,0,0],[1,0,0,1,1,0,0,0],[0,0,0,1,1,1,0,0],
    #           [0,0,0,0,1,1,1,0],[0,0,0,0,0,1,1,1],[0,0,0,0,0,0,1,1]]
    # proc_6 = [[1,1,0,0,0,0],[1,1,1,0,0,0],[0,1,1,1,0,0],[0,0,1,1,1,0],[0,0,0,1,1,1],[0,0,0,0,1,1]]
    # proc_7 = [[1,1,0,0,0,0,0,0,1,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0,0,0],[0,1,1,1,0,0,0,0,0,0,0,0,0,0],
    #           [0,0,1,1,1,0,0,0,0,0,0,0,0,0],[0,0,0,1,1,1,0,0,0,0,0,0,0,0],[0,0,0,0,1,1,1,0,0,0,0,0,0,0],
    #           [0,0,0,0,0,1,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,1,1,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,1,1,0,0,0,0],
    #           [0,0,0,0,0,0,0,0,1,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,1,1,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,1,1,1],
    #           [0,0,0,0,0,0,0,0,0,0,0,1,1,0],[0,0,0,0,0,0,0,0,0,0,0,1,0,1]]
    # astrocyte_1 = [proc_1,proc_2,proc_3,proc_4,proc_5,proc_6,proc_7]
    
    """
    soma + 1 long process
    """
#    astrocyte_1 = [[[1,1,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0],[0,1,1,1,0,0,0,0,0,0],[0,0,1,1,1,0,0,0,0,0],
#                    [0,0,0,1,1,1,0,0,0,0],[0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,1,1,1,0,0],[0,0,0,0,0,0,1,1,1,0],
#                    [0,0,0,0,0,0,0,1,1,1],[0,0,0,0,0,0,0,0,1,1]]]
    
    """
    3 outer compartments
    """
#    astrocyte_1 = [[[1,1],[1,1]]]
    
    """
    4 outer compartments
    """
    astrocyte_1 = [[[1,1,0],[1,1,1],[0,1,1]]]

    def __init__(self, path, load_values):
        self.astrocyte_1 = [[[1,1,0,0],[1,1,1,0],[0,1,1,1],[0,0,1,1]]]
        if load_values:
            self.load_config(path)
        self.save_config(path) 
        self._soma = [1] # the soma
        self.astrocyte_1.insert(0, self._soma) # adds the soma
        
    
    def save_config(self, path):
        config = ConfigParser()
        config.add_section("geometry")
        config.set("geometry","astrocyte_1",str(self.astrocyte_1))
        config.write(open(path,"w"))
        
    def load_config(self, path):
        config = ConfigParser()
        config.read(path)
        self.astrocyte_1 = ast.literal_eval(config.get("geometry","astrocyte_1"))
        
        
    def get_name(self):
        return self._name