"""
from peter
"""

from astrocyte.intracellular_space import IntracellularSpace
from astrocyte.endoplasmatic_reticulum import EndoplasmaticReticulum
from astrocyte.extracellular_space import ExtracellularSpace
from utils.writable import Writable


class ER_Ca_diff(Writable):
    
    _name = "ER_Ca_diff"
    
    # d_Ca_er = 0.001 #1/s
    # d_Ca_er = 3e-10 #1/s

    # D_Ca_er = 13e-12 # Oschmann
    # D_Ca_er = 10e-12 # Verisokin
    # D_Ca_er = 25e-12 # Bartie
    D_Ca_er = 30e-12 # Gordleeva 2019
    
    def compute_current(self, comp, connections, V, x, surface):
        from astrocyte.astrocyte import Astrocyte
        
        er_Ca_diff = 0
        for c in connections:    
            d_Ca_er = self.D_Ca_er*min(Astrocyte._d[c]._volume*Astrocyte._d[c]._ratio_er/x,surface)/(V*x)
            er_Ca_diff += d_Ca_er*(Astrocyte._d[c].er.calcium-Astrocyte._d[comp].er.calcium)
            
        return er_Ca_diff
    
    def get_name(self):
        return self._name
        
class IS_Ca_diff(Writable):
    
    _name = "IS_Ca_diff"
    
    # D_Ca = 13e-12 # Oschmann
    # D_Ca = 10e-12 # Verisokin
    # D_Ca = 25e-12 # Bartie
    D_Ca = 30e-12 # Gordleeva 2019
    
    def compute_current(self,comp, connections, V, x, surface):
        from astrocyte.astrocyte import Astrocyte
        
        is_Ca_diff = 0
        for c in connections:    
            d_Ca = self.D_Ca*min(Astrocyte._d[c]._volume*(1-Astrocyte._d[c]._ratio_er)/x,surface)/(V*x)
            is_Ca_diff += d_Ca*(Astrocyte._d[c].intra.calcium-Astrocyte._d[comp].intra.calcium)
            
        return is_Ca_diff
    
    def get_name(self):
        return self._name
    
class IS_IP3_diff(Writable):

    _name = "IS_IP3_diff"   
    
    D_ip3 = 3e-10 # Oschmann 3e-10 
    # D_ip3 = 10e-12 # Verisokin
    # D_ip3 = 280e-12 # Bartie
    
    def compute_current(self, comp, connections, V, x, surface):
        from astrocyte.astrocyte import Astrocyte
        
        is_ip3_diff = 0
        for c in connections:
            d_ip3 = self.D_ip3*min(Astrocyte._d[c]._volume*(1-Astrocyte._d[c]._ratio_er)/x,surface)/(V*x)
            is_ip3_diff += d_ip3*(Astrocyte._d[c].intra.ip3-Astrocyte._d[comp].intra.ip3)
            
        return is_ip3_diff
        
        
    def get_name(self):
        return self._name    
    
class IS_K_diff(Writable):
    
    _name = "IS_K_diff"
    
    D_K = 1.96e-9
    
    def compute_current(self, comp, connections, V, x, surface):
        from astrocyte.astrocyte import Astrocyte
        
        is_K_diff = 0
        for c in connections:
            d_K = self.D_K*min(Astrocyte._d[c]._volume*(1-Astrocyte._d[c]._ratio_er)/x,surface)/(V*x)
            is_K_diff += d_K*(Astrocyte._d[c].intra.potassium-Astrocyte._d[comp].intra.potassium)
            
        return is_K_diff
    
    def get_name(self):
        return self._name  
            
class IS_Na_diff(Writable):
    
    _name = "IS_Na_diff"
    
    D_Na = 1.33e-9
    
    def compute_current(self, comp, connections, V, x, surface):
        from astrocyte.astrocyte import Astrocyte
        
        is_Na_diff = 0
        for c in connections:
            d_Na = self.D_Na*min(Astrocyte._d[c]._volume*(1-Astrocyte._d[c]._ratio_er)/x,surface)/(V*x)
            is_Na_diff += d_Na*(Astrocyte._d[c].intra.sodium-Astrocyte._d[comp].intra.sodium)
            
        return is_Na_diff
    
    def get_name(self):
        return self._name
    
class ES_Ca_diff(Writable):
    
    _name = "ES_Ca_diff"
    
    D_Ca_es = 0.13e-10 # COMSOL
    # D_ca = 30e-6*2 # Gordleeva 2019
    
    def compute_current(self,comp, connections, V, x, surface):
        from astrocyte.astrocyte import Astrocyte
        
        es_Ca_diff = 0
        for c in connections:    
            d_Ca_es = self.D_Ca_es*min(Astrocyte._d[c]._volume*(1-Astrocyte._d[c]._ratio_er)/x,surface)/(V*x)
            es_Ca_diff += d_Ca_es*(Astrocyte._d[c].extra.calcium-Astrocyte._d[comp].extra.calcium)
            
        return es_Ca_diff
    
    def get_name(self):
        return self._name
    
class ES_K_diff(Writable):
    
    _name = "ES_K_diff"
    
    D_K_es = 1.96e-9
    
    def compute_current(self, comp, connections, V, x, surface):
        from astrocyte.astrocyte import Astrocyte
        
        es_K_diff = 0
        for c in connections:
            d_K_es = self.D_K_es*min(Astrocyte._d[c]._volume*(1-Astrocyte._d[c]._ratio_er)/x,surface)/(V*x)
            es_K_diff += d_K_es*(Astrocyte._d[c].extra.potassium-Astrocyte._d[comp].extra.potassium)
            
        return es_K_diff
    
    def get_name(self):
        return self._name
    
class ES_Na_diff(Writable):
    
    _name = "ES_Na_diff"
    
    D_Na_es = 1.33e-9
    
    def compute_current(self, comp, connections, V, x, surface):
        from astrocyte.astrocyte import Astrocyte
        
        es_Na_diff = 0
        for c in connections:
            d_Na_es = self.D_Na_es*min(Astrocyte._d[c]._volume*(1-Astrocyte._d[c]._ratio_er)/x,surface)/(V*x)
            es_Na_diff += d_Na_es*(Astrocyte._d[c].extra.sodium-Astrocyte._d[comp].extra.sodium)
            
        return es_Na_diff
    
    def get_name(self):
        return self._name

class Compartment_pathway(Writable):
    
    _name = "Compartment_pathway"
    
    def __init__(self, intra: IntracellularSpace, extra: ExtracellularSpace, er: EndoplasmaticReticulum):
        self._intra = intra
        self._extra = extra
        self._er = er
        
        self.er_Ca_d = ER_Ca_diff()
        self.is_Ca_d = IS_Ca_diff()
        self.is_IP3_d = IS_IP3_diff()
        self.is_K_d = IS_K_diff()
        self.is_Na_d = IS_Na_diff()
        self.es_Ca_d = ES_Ca_diff()
        self.es_K_d = ES_K_diff()
        self.es_Na_d = ES_Na_diff()
    
    def get_name(self):
        return self._name
    
    def compute_currents(self,x,V,surface,cons, r_er, comp):
        
        surface_er = V*r_er/x
        surface_intra = V*(1-r_er)/x
        surface_extra = V*(1-r_er)/x
        volume_er = V*r_er
        volume_intra = V*(1-r_er)
        volume_extra = V*(1-r_er)
        
        er_Ca_diff = self.er_Ca_d.compute_current(comp, cons, volume_er, x, surface_er)
        is_Ca_diff = self.is_Ca_d.compute_current(comp, cons, volume_intra, x, surface_intra)
        is_IP3_diff = self.is_IP3_d.compute_current(comp, cons, volume_intra, x,surface_intra)
        is_K_diff = self.is_K_d.compute_current(comp, cons, volume_intra, x, surface_intra)
        is_Na_diff = self.is_Na_d.compute_current(comp, cons, volume_intra, x, surface_intra)
        es_Ca_diff = self.es_Ca_d.compute_current(comp, cons, volume_extra, x, surface_extra)
        es_K_diff = self.es_K_d.compute_current(comp, cons, volume_extra, x, surface_extra)
        es_Na_diff = self.es_Na_d.compute_current(comp, cons, volume_extra, x, surface_extra)
        
        return {self.er_Ca_d.get_name(): er_Ca_diff,
                self.is_Ca_d.get_name(): is_Ca_diff,
                self.is_IP3_d.get_name(): is_IP3_diff,
                self.is_K_d.get_name(): is_K_diff,
                self.is_Na_d.get_name(): is_Na_diff,
                self.es_Ca_d.get_name(): es_Ca_diff,
                self.es_K_d.get_name(): es_K_diff,
                self.es_Na_d.get_name(): es_Na_diff}

        