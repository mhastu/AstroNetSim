"""
from lea modified by peter
"""
from utils.writable import Writable

class EndoplasmaticReticulum(Writable):
    calcium = 19.63e-3

    _name = "EndoplasmaticReticulum"

    def __init__(self, w1):
        self._w1 = w1 # S*r_er^2/(V*F*r_er)

    def change_in_calcium_ode(self, I_ip3, I_serca, I_caleak, I_ca_er_diff):
        return self._w1 * (-I_ip3 + I_serca - I_caleak) + I_ca_er_diff


    def get_name(self):
        return self._name
