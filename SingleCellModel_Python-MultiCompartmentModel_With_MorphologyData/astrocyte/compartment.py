"""
from lea modified by peter
"""
import math
# from math import exp
from configparser import ConfigParser

import numpy as np

import constants
from astrocyte.endoplasmatic_reticulum import EndoplasmaticReticulum
from astrocyte.extracellular_space import ExtracellularSpace
from astrocyte.glut_pathway import GluTPathway
from astrocyte.intracellular_space import IntracellularSpace
from astrocyte.ip3_production import IP3Production
from astrocyte.mglur_pathway import MGluRPathway
from astrocyte.compartment_pathway import Compartment_pathway
from utils.writable import Writable
from utils import config_writer

class Compartment(Writable):
    
    # calcium_tot = 1.82073  # tot Ca^2+ in this compartment
    # sodium_tot = 165  # tot Na^+ in this compartment
    
    # potassium_tot = 103  # tot K^+ in this compartment

    membrane_capacitance = 0.01  # TODO does this belong here.

    voltage = -85.88e-3  # Membrane Voltage
    
    # _x = 10e-6 # comp length in m

    _name = "Compartment"

    def __init__(self, r,  comp_nr, process_nr, lvl, connections, length, r_er):
        self._x = length # comp length in m
        self._r = r # radius
        self._comp_connections = [] #list of connected  compartments
        self._comp_nr = comp_nr # compartment number
        self._process_nr = process_nr # process number
        self._comp_lvl = lvl # compartment lvl(how far away from soma, defines radius)
        self._connections = connections # help to get _comp_connections
        
        self._surface = self.get_surface() # surface for connections
        self._volume = self.get_volume() # compartment volume
        self._svr = self.get_svr() # surface to volume ratio
        # ratio_er = self.get_ratio_er() # ratio of the ER to intra volume
        self._ratio_er = r_er
        sqrt = math.sqrt(self._ratio_er)
        
        self.er = EndoplasmaticReticulum(self._svr * sqrt / (constants.FARADAY_CONSTANT * self._ratio_er))
        self.extra = ExtracellularSpace(self._svr / (constants.FARADAY_CONSTANT * (1 - self._ratio_er)))

        self.intra = IntracellularSpace(self._svr / (constants.FARADAY_CONSTANT * (1 - self._ratio_er)),
                                        self._svr * sqrt / (constants.FARADAY_CONSTANT * (1 - self._ratio_er)))

        self.gluT_pathway = GluTPathway(self.intra, self.extra)
        self.ip3_production = IP3Production(self.intra, self.extra)
        self.mglur_pathway = MGluRPathway(self.intra, self.er, (1 - self._ratio_er) / (self._svr * sqrt))
        self.compartment_pathway = Compartment_pathway(self.intra, self.extra, self.er)

        self._interpolation_f = None

    def get_name(self):
        return self._name
    
    # def get_ratio_er(self):
    #     """
    #     Computes the volume ratio of the ER in comparison to the volume of the intracellular space. Depends on SVR (Surface Volume Ratio).
    #     The CLOSER to the Soma, the LARGER this value. Compartments close to the soma should have a ratio of about 0.15 (SVR 38e6)
    #     The FURTHER AWAY from the soma, the SMALLER (0) this value
    #     Reason: Astrocytic compartments close to the synapse do not contain an internal Ca^2+ store.
    #     :return:
    #         Volume Ratio ER / Intra Cellular Space.
    #     """
    #     return 0.15 * exp(-0.002 * (self._svr / 1e6) ** 2.32)
    
    def get_surface(self): #surf of comp to comp
        return self._r**2*np.pi 
    
    def get_volume(self): #volume of compartment
        return self._r**2*np.pi*self._x

    def get_svr(self): # get surface to volume ratio
        return (self._r*2*np.pi*self._x)/(self._volume)
    
    def to_file(self, path): # save values in config
       config_parser = ConfigParser()
       config_writer.write_recursively(self, config_parser)

       with open(path, 'w') as f:
           config_parser.write(f)
          
    def from_file(self, path): # load values from config
        config_parser = ConfigParser(inline_comment_prefixes="#")
        config_parser.read(path)
        config_writer.read_recursively(self, config_parser)