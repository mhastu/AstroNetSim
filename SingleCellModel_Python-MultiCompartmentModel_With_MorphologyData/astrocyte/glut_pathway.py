"""
from lea
"""
from math import log
from math import exp

import constants
import simulation_config
from astrocyte.extracellular_space import ExtracellularSpace
from astrocyte.intracellular_space import IntracellularSpace
from utils.writable import Writable


class NKA(Writable):
    _name = "NKA"
    max_current = 1.52  # maximal pumping current of NKA
    half_saturation_sodium = 10  # [mM]
    half_saturation_potassium = 1.5  # [mM]

    def compute_current(self, intra_sodium, extra_potassium):
        """
        Equation 12, Oschmann 2017
        NKA: Na^+/K^+-ATPase. Transport AGAINST concentration gradient
        (Nobel price in chemistry =) )
        Na^+ out and 2 K^+ into (simplified) [luo1994]

        :return:
            NKA Current
        """
        sodium_exp = intra_sodium ** 1.5
        nka_sodium = sodium_exp / (sodium_exp + self.half_saturation_sodium ** 1.5)
        nka_potassium = extra_potassium / (extra_potassium + self.half_saturation_potassium)
        return self.max_current * nka_sodium * nka_potassium

    def get_name(self):
        return self._name


class NCX(Writable):
    _name = "NCX"
    max_current = 0.001  # maximal pump current of NCX
    half_saturation_sodium = 87.5
    half_saturation_calcium = 1.380
    k_saturation = 0.1  # saturation factor, ensuring saturation at large negative potentials
    eta = 0.35  # TODO ???

    def __init__(self, inv_thermal_voltage):
        self._inv_thermal_voltage = inv_thermal_voltage

    def get_name(self):
        return self._name

    def compute_current(self, intra_sodium, extra_sodium, intra_calcium, extra_calcium, voltage):
        extra_sodium_exp = extra_sodium ** 3
        intra_sodium_exp = intra_sodium ** 3

        weird_exponential = exp(self.eta * voltage * self._inv_thermal_voltage)
        weird_exponential_other = exp((self.eta - 1) * voltage * self._inv_thermal_voltage)

        # TODO: Potenz 2?
        ncx_sodium = extra_sodium_exp / (self.half_saturation_sodium ** 3 + extra_sodium_exp)
        ncx_calcium = extra_calcium / (self.half_saturation_calcium + extra_calcium)

        last_weird_term = intra_sodium_exp / extra_sodium_exp * weird_exponential - intra_calcium / extra_calcium * weird_exponential_other
        nenner = 1. / (1 + self.k_saturation * weird_exponential_other)

        return self.max_current * ncx_sodium * ncx_calcium * last_weird_term * nenner

class GluT(Writable):
    _name = "GluT"
    max_current = 0.75  # maximal transport current of glutamate transporter
    half_saturation_sodium = 15
    half_saturation_potassium = 5  # not experimentally known. close to concentration in extra cellular space in the beginning. NOT KNOWN.
    half_saturation_glutamate = 34e-3

    def get_name(self):
        return self._name

    def compute_current(self, intra_potassium, extra_sodium, extra_glu):
        """
        Get 1 Glu^- to intra. space using glutamate transport. Co-Transports
        3 Na^+ into intra. space and 1 K^+ out of intra. space.
        Additional transport of H^+ and more Glu^- is ignored but their charges are considered in V computation. (???)
        Glu is not needed anywhere else. -> We don't keep track of it

        :return:
            Glu Current
        """
        gluT_potassium = intra_potassium / (intra_potassium + self.half_saturation_potassium)
        sodium_exp = extra_sodium ** 3
        gluT_sodium = sodium_exp / (sodium_exp + self.half_saturation_sodium ** 3)
        gluT_glutamate = extra_glu / (extra_glu + self.half_saturation_glutamate)
        return self.max_current * gluT_potassium * gluT_sodium * gluT_glutamate


class Leak(Writable):

    conductance = 1

    def __init__(self, conductance, thermal_voltage, _name="leak"):
        self.conductance = conductance
        self._thermal_voltage = thermal_voltage
        self._name = _name

    def nernst_potential(self, outer_value, inner_value):
        """
        Compute nernst potential, the electric potential stemming from difference in particle concentration
        :param outer_value: Concentration of element in extra cellular space
        :param inner_value: Concentration of element in intra cellular space
        :return:
            Nernst Potential
        """
        return self._thermal_voltage * log(outer_value / inner_value)

    def compute_current(self, extra_concentration, intra_concentration, voltage):
        """
        URI :)
        :param extra_concentration:
        :param intra_concentration:
        :param voltage:
        :return:
        """
        return self.conductance * (voltage - self.nernst_potential(extra_concentration, intra_concentration))

    def get_name(self):
        return self._name


class GluTPathway(Writable):
    """
    Describes the GluT driven transport of Ca^2+ between extracellular and intracellular spaces of a compartment.
    Includes a NCX (Sodium-Calcium-Exchanger), A GluT transport, NKA (Sodium, Potassion-ATPase) and a sodium / potassium leak
    """

    # _sodium_conductance = 6.5
    _sodium_conductance = 13
    # _potassium_conductance = 79.1
    _potassium_conductance = 162.46

    _name = "GluTPathway"

    def __init__(self, intra: IntracellularSpace, extra: ExtracellularSpace):
        self._intra = intra
        self._extra = extra

        thermal_voltage = constants.MOLAR_GAS_CONSTANT * simulation_config.TEMPERATURE / constants.FARADAY_CONSTANT  # RT / F
        inv_thermal_voltage = 1. / thermal_voltage  # for speed

        self.ncx = NCX(inv_thermal_voltage)
        self.nka = NKA()
        self.gluT = GluT()

        self.potassium_leak = Leak(self._potassium_conductance, thermal_voltage, _name="PotassiumLeak")
        self.sodium_leak = Leak(self._sodium_conductance, thermal_voltage, _name="SodiumLeak")


    def compute_currents(self, voltage):

        ncx_current = self.ncx.compute_current(self._intra.sodium, self._extra.sodium, self._intra.calcium, self._extra.calcium, voltage)
        nka_current = self.nka.compute_current(self._intra.sodium, self._extra.potassium)
        gluT_current = self.gluT.compute_current(self._intra.potassium, self._extra.sodium, self._extra.glutamate)

        potassium_leak_current = self.potassium_leak.compute_current(self._extra.potassium, self._intra.potassium, voltage)
        sodium_leak_current = self.sodium_leak.compute_current(self._extra.sodium, self._intra.sodium, voltage)

        return {self.ncx.get_name(): ncx_current,
                self.nka.get_name(): nka_current,
                self.gluT.get_name(): gluT_current,
                self.sodium_leak.get_name(): sodium_leak_current,
                self.potassium_leak.get_name(): potassium_leak_current}


    def get_name(self):
        return self._name
