"""
from lea
"""
import math

from astrocyte.extracellular_space import ExtracellularSpace
from astrocyte.intracellular_space import IntracellularSpace
from utils.writable import Writable


class IP3Production(Writable):

    _name = "IP3Production"
    max_production_by_plc_beta = 0.1e-3  # \nu_beta
    glu_plc_beta_affinity = 1.3e-3
    plc_beta_inhibition = 10e-3  # inhibits ca
    ca_plc_beta_affinity = 0.6e-3

    max_production_by_plc_delta = 0.02e-3  # \nu_delta
    plc_delta_inhibition = 1.5e-3  # kappa_delta
    ca_plc_delta_affinity = 0.1e-3  # K_Plcdelta

    ca_ip33k_affinity = 0.7e-3  # K_D
    ip3_ip33k_affinity = 1e-3  # K_3
    max_degradation_by_ip33k = 2e-3  # nu_3k

    max_degradation_by_ip5p = 0.04  # r_5p

    def __init__(self, intra: IntracellularSpace, extra: ExtracellularSpace):
        self._intra = intra
        self._extra = extra

    def compute_prod_plc_beta(self, intra_calcium, extra_glu):
        glu_exp = math.pow(extra_glu, 0.7)
        glu_affinity = (self.glu_plc_beta_affinity + self.plc_beta_inhibition * intra_calcium / (intra_calcium + self.ca_plc_beta_affinity))
        return self.max_production_by_plc_beta * glu_exp / (glu_exp + math.pow(glu_affinity, 0.7))

    def compute_prod_plc_delta(self, intra_calcium, intra_ip3):
        production_term = self.max_production_by_plc_delta / (1 + intra_ip3 / self.plc_delta_inhibition)

        calcium_exp = intra_calcium ** 2
        affinity_term = calcium_exp / (calcium_exp + self.ca_plc_delta_affinity ** 2)
        return production_term * affinity_term

    def compute_deg_ip3_3k(self, intra_calcium, intra_ip3):
        calcium_exp = intra_calcium ** 4
        calcium_term = (calcium_exp / (calcium_exp + self.ca_ip33k_affinity**4))
        ip3_term = (intra_ip3 / (intra_ip3 + self.ip3_ip33k_affinity))

        return self.max_degradation_by_ip33k * calcium_term * ip3_term

    def compute_deg_ip_5p(self, intra_ip3):
        return self.max_degradation_by_ip5p * intra_ip3

    def compute_measures(self):
        measures = {"PLCBeta": self.compute_prod_plc_beta(self._intra.calcium, self._extra.glutamate),
                    "PLCDelta": self.compute_prod_plc_delta(self._intra.calcium, self._intra.ip3),
                    "IP33K": self.compute_deg_ip3_3k(self._intra.calcium, self._intra.ip3),
                    "IP5P": self.compute_deg_ip_5p(self._intra.ip3)}
        return measures


    def get_name(self):
        return self._name