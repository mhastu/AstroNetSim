"""
from peter
"""
import numpy as np
# import math
from scipy.integrate import solve_ivp
from scipy.interpolate import interpolate

from astrocyte.geometry import Geometry
from astrocyte.compartment import Compartment
from utils.writable import Writable
import json
from configparser import ConfigParser
from utils import config_writer


class Astrocyte(Writable):
    
    _name = "Astrocyte"
    
    _d = {}
    _comp_nr = 1
    svr = 0
    #r_list = [1e-6]
    r_list = [4.09333333e-06, 5.02526250e-06, 4.38460750e-06, 3.81073750e-06, 1.95486750e-06]
    #r_list = [8e-6,4e-6,3e-6,2e-6,1e-6,.5e-6,.25e-6] # compartment radius in m
    # r_list = [1e-6,3e-6,4e-6] # radius in m
    #r_list = [8e-6,4e-6,3e-6] # radius in m
    # r_list = [8e-6,20.1e-8,18.5e-8,16.9e-8,15.5e-8,14.3e-8,13.2e-8] # radius in m
    #r_er_list = [.15]
    r_er_list = [0.14966032, 0.14972326, 0.14968286, 0.14963516, 0.14928962]
    # r_er_list = [.07,.08,.09]
    #r_er_list = [.15,.10,.09]
    #r_er_list = [.15,.1,.09,.08,.07,.06,.05] 
    # r_er_list = [.05,.06,.07,.08,.09,.1,.15] 
    #x = [10e-6,10e-6,10e-6,10e-6,10e-6,10e-6] # comp length in m
    x = [1.228e-05, 8.632816581773566e-07, 8.605610088773283e-07, 1.2588612711494655e-06, 6.879370602751388e-06]
    
    """
    selects what to plot:
    0 = intra calcium, 1 = ER calcium , 2 = extra calcium, 3 = h,
    4 = ip3, 5 = intra sodium, 6 = extra sodium, 7 = intra potassium,
    8 = extra potassium, 9 = membrane voltage
    """
    y_plot = 0
    
    dummy_comp = Compartment(1, 1, 1, 1, [], 1,  0.5) # just used to save all values in the ini files
    
    # _spike_train = 2e-6 * np.ones(3000) #og
    # _spike_train[625:1250] = 2e-3 #og
    # _spike_train = 2e-6 * np.ones(600) # glutamate stimulus lea
    #_spike_train = 2e-6 * np.ones(3000) # glutamate stimulus pitta
    # _spike_train[125:250] = 1e-3 # glutamate stimulus lea
    # _spike_train[375:500] = 6e-3 # glutamate stimulus lea
    # _spike_train[625:1250] = 1e-3 # glutamate stimulus pitta
    # _spike_train[1875:2500] = 6e-3 # glutamate stimulus pitta

    _spike_train = 2e-6 * np.ones(5000) #5000
    _spike_train[500:1125] = 0.1e-3  # glutamate stimulus anna 0.1 [500:1125] in mili Mol
    _spike_train[2000:2625] = 1e-3  # glutamate stimulus anna
    _spike_train[3500:4125] = 6e-3  # glutamate stimulus anna
    # print(_spike_train.shape)
    _spike_train = list(_spike_train)
    
    time_start = 0
    time_end = 500 #500
    # _spike_comps = {"compartment_0": _spike_train}
    _spike_comps = {"compartment_20": _spike_train}
    #_spike_comps = {"compartment_15": _spike_train, "compartment_18": _spike_train, "compartment_2": _spike_train, "compartment_23": _spike_train, "compartment_26": _spike_train,"compartment_28": _spike_train, "compartment_37": _spike_train, "compartment_40": _spike_train, "compartment_42": _spike_train, "compartment_43": _spike_train, "compartment_44": _spike_train, "compartment_45": _spike_train, "compartment_53": _spike_train, "compartment_55": _spike_train, "compartment_57": _spike_train}
    #_spike_comps = {"compartment_1": _spike_train,"compartment_5": _spike_train,"compartment_6": _spike_train,"compartment_11": _spike_train,"compartment_20": _spike_train,"compartment_24": _spike_train,"compartment_25": _spike_train,"compartment_27": _spike_train,"compartment_35": _spike_train}
    
    def __init__(self, geometry_path, spike_path, values_path, load_values, Morph=False, svr=0, length=0, connection_matrix=[]):
        if load_values: # load values and stimulated compartments from ini files
            self.from_file(values_path)
            self.load_spike(spike_path)
        if Morph:
            self.r_list = (1/np.array(svr))*4/2
            self.r_list[0] = (1/np.array(svr)[0])*3
            self.r_er_list = 0.15 * np.exp(-0.002 * (np.array(svr) * 1e-6)) ** 2.32
            self.x = length
            self._astrocyte_1 = connection_matrix
        else:
            self._geometry = Geometry(geometry_path, load_values) # create/load geometry of the astrocyte
            self._astrocyte_1 = self._geometry.astrocyte_1
        self.get_astrocyte()
        self.save_spike(spike_path) # save stimulated compartments in ini files
        
    def get_name(self):
        return self._name
    
    def get_astrocyte(self):
        # print(self._astrocyte_1)
        for process_nr, process in enumerate(self._astrocyte_1): # iterate over processes
            self._process_nr = process_nr
            # print('process',process_nr, process)
            if np.any(process): 
                if len(process) == 1: # if just one compartment connected to soma, otherwise error
                    n_of_comps = 1
                    connections = [0]
                    comp_lvl = [0]
                    lvl = [0]
                else: 
                    n_of_comps = len(process) # number of compartments in this process
                    con_matrix = process - np.eye(n_of_comps) #connection matrix
                    connections = [np.where(con_matrix[i] == 1)[0] for i in range(n_of_comps)] #help for list of connections 
                    comp_lvl = [[j+1] for j,con in enumerate(connections)]
                    comp_lvl[-1] = []
                    #comp_lvl = [[i for i in con if i>j]for j,con in enumerate(connections)] 
                    
                    # lvl = np.zeros_like(comp_lvl, dtype=object) #original
                    lvl = (np.zeros_like(np.array(comp_lvl, dtype=object))) #anna
                    # lvl = np.zeros(np.array(comp_lvl), dtype=object)
                    # lvl = np.array(comp_lvl)*0 #anna
                connections = [i+self._comp_nr for i in connections] # correction for ascending comp_number
                    
                if process_nr == 0: # the soma
                    self._d["compartment_0"] = self.get_compartment(0, 0, [])
                    continue
                
                for j,comps in enumerate(comp_lvl):
                    if j == 0:
                        # if np.any(np.array(comp_lvl, dtype=object)): #og
                        #     lvl[0] = comp_lvl[0] #og
                        for element in comp_lvl:
                            if np.any(element): #anna
                                lvl[0] = comp_lvl[0] #anna
                    else:
                        if np.any(lvl[j-1]):
                            for comp in lvl[j-1]:
                                if type(lvl[j]) == list:
                                    lvl[j] += comp_lvl[comp]
                                else:
                                    lvl[j] = comp_lvl[comp]
                        else:
                            lvl = lvl[:j-1] # remove unnecessary values
                            break
                if np.any(lvl):
                    lvl = np.concatenate(([0],lvl)) 
                    lvl[0] = [0] # at what lvl the compartments are(for the radius)
                            
                if len(lvl) == 1: # if just one compartment connected to soma, otherwise error
                        self._d[f"compartment_{j + self._comp_nr}"] = self.get_compartment(j + self._comp_nr, 0, [])
                else:
                    for i,l in enumerate(lvl):
                        # print(l)
                        for j in l:
                            compartment_name = j + self._comp_nr #anna
                            # print(f"compartment_{compartment_name}", i, connections[j])
                            self._d[f"compartment_{compartment_name}"] = self.get_compartment(j + self._comp_nr, i, connections[j])
                # start changes
                self._comp_nr += len(process)
                
            
    def get_compartment(self, comp_nr, comp_lvl, connections): # create single compartments        

        if self._process_nr != 0:
            comp_lvl += 1 # if its not the soma    
        if comp_lvl < len(self.r_list)-1: 
            return Compartment(self.r_list[comp_lvl], comp_nr, self._process_nr, comp_lvl, connections, self.x[comp_lvl], self.r_er_list[comp_lvl])
        else: # for longer processes the radius doesnt get smaller than the last list entry
            return Compartment(self.r_list[-1], comp_nr, self._process_nr, comp_lvl, connections, self.x[-1], self.r_er_list[-1])

    def save_spike(self, path):
        config = ConfigParser()
        config["spike_train"] = self._spike_comps
        
        with open(path, 'w') as f:
            config.write(f)
        
    def load_spike(self, path):
        config = ConfigParser()
        config.read(path)
        for key in config["spike_train"]:
            self._spike_comps[key] = json.loads(config.get("spike_train",key))

    """
    from lea modified by peter
    """

    def solve(self):
        
        initial_values = [[self._d[comp].intra.calcium, self._d[comp].er.calcium, self._d[comp].extra.calcium, self._d[comp].mglur_pathway.ip3.h,
                           self._d[comp].intra.ip3, self._d[comp].intra.sodium, self._d[comp].extra.sodium, self._d[comp].intra.potassium,
                           self._d[comp].extra.potassium, self._d[comp].voltage] for comp in self._d if "compartment" in comp]
        initial_values = [i for lst in initial_values for i in lst] # flat the initial_values
        
        t = np.linspace(self.time_start, self.time_end, len(next(iter(self._spike_comps.values())))) # The length of the first spike value
        for comp in self._spike_comps:
            if comp in self._d:
                self._d[comp]._interpolation_f = interpolate.interp1d(t, self._spike_comps[comp])
        sol = solve_ivp(self.dxdt, (self.time_start, self.time_end), initial_values, method='Radau', t_eval=t)  # for some reason, higher order RK fails?
        
        return sol

    def dxdt(self, t, x):
        """

        :param x: x(0) internal calcium, x(1) er calcium, x(2) extra calcium,  x(3) h, x(4) internal ip3, x(5) internal sodium, x(6) external sodium,
        x(7) internal potassium, x(8) external potassium, x(9) membrane voltage
        :param t: time steps to evaluate
        :return:

        """

        len_x = len(x)//10
        x = np.reshape(x,(len_x,10))
            
        self.update(x, t)
        odes=[]
        for comp in self._d:
            if "compartment" in comp:
                mglur_currents = self._d[comp].mglur_pathway.compute_currents()
                gluT_currents = self._d[comp].gluT_pathway.compute_currents(self._d[comp].voltage)
                ip3_measures = self._d[comp].ip3_production.compute_measures()
                comp_currents = self._d[comp].compartment_pathway.compute_currents(self._d[comp]._x, self._d[comp]._volume,self._d[comp]._surface, 
                                                                                   self._d[comp]._comp_connections, self._d[comp]._ratio_er, comp)
        
                odes.extend([
                    self._d[comp].intra.change_in_calcium_ode(gluT_currents["NCX"], mglur_currents["IP3"], mglur_currents["SERCAPump"],
                                                              mglur_currents["CaLeak"], comp_currents["IS_Ca_diff"]),
                    self._d[comp].er.change_in_calcium_ode(mglur_currents["IP3"], mglur_currents["SERCAPump"], mglur_currents["CaLeak"],
                                                           comp_currents["ER_Ca_diff"]),
                    self._d[comp].extra.change_in_calcium_ode(gluT_currents["NCX"], comp_currents["ES_Ca_diff"]),
                    self._d[comp].mglur_pathway.ip3.change_in_h_ode(self._d[comp].intra.ip3, self._d[comp].intra.calcium),
                    self._d[comp].intra.change_in_ip3_ode(ip3_measures["PLCBeta"], ip3_measures["PLCDelta"], ip3_measures["IP33K"],
                                                          ip3_measures["IP5P"], comp_currents["IS_IP3_diff"]),
                    self._d[comp].intra.change_in_sodium_ode(gluT_currents["GluT"], gluT_currents["NKA"], gluT_currents["NCX"],
                                                             gluT_currents["SodiumLeak"], comp_currents["IS_Na_diff"]),
                    self._d[comp].extra.change_in_sodium_ode(gluT_currents["GluT"], gluT_currents["NKA"], gluT_currents["NCX"],
                                                             gluT_currents["SodiumLeak"], comp_currents["ES_Na_diff"]),
                    self._d[comp].intra.change_in_potassium_ode(gluT_currents["GluT"], gluT_currents["NKA"], gluT_currents["PotassiumLeak"],
                                                                comp_currents["IS_K_diff"]),
                    self._d[comp].extra.change_in_potassium_ode(gluT_currents["GluT"], gluT_currents["NKA"], gluT_currents["PotassiumLeak"],
                                                                comp_currents["ES_K_diff"]),
                    self.change_in_voltage_ode(mglur_currents["IP3"], mglur_currents["SERCAPump"], mglur_currents["CaLeak"], gluT_currents["NCX"],
                                               gluT_currents["GluT"], gluT_currents["NKA"], gluT_currents["SodiumLeak"], gluT_currents["PotassiumLeak"], comp)
                                               ])
            else:
                continue

        return odes

    def update(self, x, t):
        epsilon = 0.0000001

        
        for i, comp in enumerate(self._d):
            if "compartment" in comp:
                if comp in self._spike_comps:
                    self._d[comp].extra.glutamate = self._d[comp]._interpolation_f([t])[0]
                self._d[comp].intra.calcium = max(x[i][0], epsilon)
                self._d[comp].er.calcium = max(x[i][1], epsilon)
                self._d[comp].extra.calcium = max(x[i][2], epsilon)
                self._d[comp].mglur_pathway.ip3.h = x[i][3]
                self._d[comp].intra.ip3 = x[i][4]
        
                self._d[comp].intra.sodium = max(x[i][5], epsilon)
                self._d[comp].extra.sodium = max(x[i][6], epsilon)
        
                self._d[comp].intra.potassium = max(x[i][7], epsilon)
                self._d[comp].extra.potassium = max(x[i][8], epsilon)
        
                self._d[comp].voltage = x[i][9]
                # self.validate(comp)
                

    # def validate(self, comp):
    #     """
    #     Ensure that Ca/Na/K values are consistent.
    #     :return:
    #     """
    #     assert math.isclose(self._d[comp].calcium_tot - self._d[comp].intra.calcium - self._d[comp].er.calcium, self._d[comp].extra.calcium), "CA (intra,ER,extra) do not add up to tot"
    #     assert math.isclose(self._d[comp].sodium_tot - self._d[comp].intra.sodium, self._d[comp].extra.sodium), "Na (intra,extra) do not add up to tot"
    #     assert math.isclose(self._d[comp].potassium_tot - self._d[comp].intra.potassium, self._d[comp].extra.potassium), "K (intra, extra) do not add up to tot"

    def change_in_voltage_ode(self, I_ip3, I_serca, I_caleak, I_ncx, I_gluT, I_nka, I_naleak, I_kleak, comp):
        # return - 1. / self.membrane_capacitance * (- 2 * I_ip3 + 2 * I_serca - 2 * I_caleak + I_ncx - 2 * I_gluT + I_nka + I_naleak + I_kleak)
        return - 1. / self._d[comp].membrane_capacitance * (I_ncx - 2 * I_gluT + I_nka + I_naleak + I_kleak)

    def to_file(self, path): # save values in config
       config_parser = ConfigParser()
       config_writer.write_recursively(self, config_parser)

       with open(path, 'w') as f:
           config_parser.write(f)
          
    def from_file(self, path): # load values from config
        config_parser = ConfigParser(inline_comment_prefixes="#")
        config_parser.read(path)
        config_writer.read_recursively(self, config_parser)
