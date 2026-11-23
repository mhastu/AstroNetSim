"""
from lea modified by peter
"""
from utils.writable import Writable


class IntracellularSpace(Writable):
    calcium = 0.073e-3
    sodium = 15
    potassium = 100

    ip3 = 0.15659e-3


    _name = "IntracellularSpace"

    def __init__(self, w1, w2):
        self._w1 = w1 # S/(V*F*(1-r_er))
        self._w2 = w2 # S*r_er^2/(V*F*(1-r_er))


    def get_name(self):
        return self._name

    def change_in_calcium_ode(self, I_ncx, I_ip3, I_serca, I_caleak, I_Ca_diff):
        return self._w1 * 0.5 * I_ncx + self._w2 * (I_ip3 - I_serca + I_caleak) + I_Ca_diff

    def change_in_potassium_ode(self, I_gluT, I_nka, I_leak, I_K_diff):
        return self._w1 * (-I_gluT + 2 * I_nka - I_leak) + I_K_diff

    def change_in_sodium_ode(self, I_gluT, I_nka, I_ncx, I_leak, I_Na_diff):
        return self._w1 * (3 * I_gluT - 3 * I_nka - 3 * I_ncx - I_leak) + I_Na_diff

    @staticmethod
    def change_in_ip3_ode(plc_beta, plc_delta, ip33k, ip5p, ip3_diff):
        return plc_beta + plc_delta - ip33k - ip5p + ip3_diff
