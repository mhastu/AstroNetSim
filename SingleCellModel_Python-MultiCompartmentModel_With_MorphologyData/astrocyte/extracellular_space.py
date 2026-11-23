"""
from lea modified by peter
"""

from utils.writable import Writable


class ExtracellularSpace(Writable):


    _name = "ExtracellularSpace"
    calcium = 1.8  # Ca^2+
    sodium = 150  # Na
    potassium = 3 # K

    glutamate = 0
    
    def __init__(self, w1):
        self._w1 = w1 # S/(V*F*(1-r_er))

    def get_name(self):
        return self._name

    def change_in_calcium_ode(self, I_ncx, I_Ca_diff):
        return self._w1 * 0.5 * -I_ncx + I_Ca_diff

    def change_in_potassium_ode(self, I_gluT, I_nka, I_leak, I_K_diff):
        return self._w1 * (I_gluT - 2 * I_nka + I_leak) + I_K_diff

    def change_in_sodium_ode(self, I_gluT, I_nka, I_ncx, I_leak, I_Na_diff):
        return self._w1 * (3 * -I_gluT + 3 * I_nka + 3 * I_ncx + I_leak) + I_Na_diff
