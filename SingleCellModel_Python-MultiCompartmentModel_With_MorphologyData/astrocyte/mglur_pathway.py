"""
from lea
"""
import constants
from astrocyte.endoplasmatic_reticulum import EndoplasmaticReticulum
from astrocyte.intracellular_space import IntracellularSpace
from utils.writable import Writable


class SERCAPump(Writable):
    max_ca_uptake = 4.5e-3  # nu_ER
    ca_affinity = 0.1e-3  # K_ER

    _name = "SERCAPump"

    def __init__(self, normalization_constant):
        """
        :param normalization_constant:  F * Vol / A
        """
        self._normalization_constant = normalization_constant


    def get_name(self):
        return self._name


    def compute_current(self, intra_calcium):
        """
        Serca Pump. Ca into ER[Li1994]
        :param intra_calcium:
        :return:
            Ca^"+ Current mediated by SERCA Pump.
        """
        calcium_squared = intra_calcium ** 2
        return self._normalization_constant * self.max_ca_uptake * calcium_squared / (calcium_squared + self.ca_affinity ** 2)


class CaLeak(Writable):
    """
    In comparison to "normal" leaks (e.g. GluT Pathway), this leak does not depend on the nernest potentials
    TODO Why?
    """

    leak_rate = 0.5 * 0.11  # r_L

    _name = "CaLeak"

    def __init__(self, normalization_constant):
        self._normalization_constant = normalization_constant  # F * Vol / A

    def get_name(self):
        return self._name

    def compute_current(self, intra_calcium, er_calcium):
        """
        Ca Leak out of CA [Li1994]
        :param intra_calcium:
        :param er_calcium:
        :return:
        """
        return self._normalization_constant * self.leak_rate * (er_calcium - intra_calcium)


class IP3(Writable):
    dissociation_subunit_1 = 0.13e-3  # what's that exactly? d_1
    dissociation_subunit_2 = 1.049e-3  # d_2
    dissociation_subunit_3 = 0.9434e-3  # d_3
    dissociation_subunit_5 = 0.08234e-3  # d_5
    max_calcium_rate = 0.5 * 6  # r_c

    h = 0.7892  # Fraction h of activated IP3 receptor channels

    ip3r_binding_rate = 0.2e3  # a_2

    _name = "IP3"

    def get_name(self):
        return self._name

    def __init__(self, normalization_constant):
        self._normalization_constant = normalization_constant  # F * Vol / A

    def compute_current(self, inter_ip3, inter_calcium, er_calcium):
        """
        Ca current through IP3 receptor channel. Ca out of ER [Li1994]
        :param inter_ip3:
        :param inter_calcium:
        :param er_calcium:
        :return:
        """
        ip3_term = (inter_ip3 / (inter_ip3 + self.dissociation_subunit_1)) ** 3
        calcium_term = (inter_calcium / (inter_calcium + self.dissociation_subunit_5)) ** 3  # ^3 -> 3 channels?
        open_probability = ip3_term * calcium_term * self.h ** 3
        return self._normalization_constant * self.max_calcium_rate * open_probability * (er_calcium - inter_calcium)

    def change_in_h_ode(self, intra_ip3, intra_calcium):
        return self.ip3r_binding_rate * (self.dissociation_subunit_2 * (intra_ip3 + self.dissociation_subunit_1) / (
                    intra_ip3 + self.dissociation_subunit_3) * (1 - self.h) - self.h * intra_calcium)


class MGluRPathway(Writable):

    _name = "MGluRPathway"

    def __init__(self, intra: IntracellularSpace, er: EndoplasmaticReticulum, wconstant):
        self._intra = intra
        self._er = er

        normalization_constant = constants.FARADAY_CONSTANT * wconstant
        self.ip3 = IP3(normalization_constant)
        self.ca_leak = CaLeak(normalization_constant)
        self.serca = SERCAPump(normalization_constant)

    def compute_currents(self):
        """
        Voltage independent!
        :return:
            dictionary containing the currents IP3, CaLeak, SERCAPump
        """
        ip3_current = self.ip3.compute_current(self._intra.ip3, self._intra.calcium, self._er.calcium)
        ca_leak_current = self.ca_leak.compute_current(self._intra.calcium, self._er.calcium)
        serca_current = self.serca.compute_current(self._intra.calcium)
        return {self.ip3._name: ip3_current, self.ca_leak._name: ca_leak_current, self.serca._name: serca_current}



    def get_name(self):
        return self._name
