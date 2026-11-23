"""
from lea modified by peter
"""
# import time

import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
from scipy import signal as sig
import time
start = time.time()
start_proc = time.process_time()

# from astrocyte.geometry import Geometry
from astrocyte.astrocyte import Astrocyte
from astrocyte.connections import Connections
# from astrocyte.compartment import Compartment

from Astrocyte_Morphology_Model import AstrocyteMorphologyModel


if __name__ == "__main__":
    
    astrocyte_model = AstrocyteMorphologyModel()
    path_astro = os.path.join(os.path.dirname(__file__), "output", "morph.ini")
    astrocyte_model.from_file(path_astro)
    [svr, length, connection_matrix, processes_array] = astrocyte_model.loading_astrocyte_data()
    
    # to find length of each main branch
    print('Number of Processes',len(processes_array))
    # for index,value in enumerate(processes_array):
    #     print('Processes',index+1,'len:',len(value))
    
    # print(np.array(svr).shape, np.array(length).shape, len(processes_array), len(connection_matrix))
    # print('svr',svr)
    # print('length',length)
    # print('processes_array',processes_array)
    # print('connection_matrix',(connection_matrix))
    
    geometry_path = os.path.join(os.path.dirname(__file__), "output", "geometry.ini")  # relative path for geometry 
    spike_path = os.path.join(os.path.dirname(__file__), "output", "spike.ini") # relative path for spike train
    values_path = os.path.join(os.path.dirname(__file__), "output", "test.ini")  # relative path for value config    
    load_values = False  #load from ini file? 
    simulate_morpology_model = False

    astrocyte_1 = Astrocyte(geometry_path, spike_path, values_path, load_values,
                            simulate_morpology_model, svr, length, connection_matrix) # create the astrocyte without the connections
    diction = astrocyte_1._d # to look at the values for debuging
    connections = Connections(astrocyte_1._d) # the connections between the compartments
    astrocyte_1._d["compartment_0"].to_file(values_path) # save values
    
    if load_values:
        for comp in astrocyte_1._d: # load config values in every compartment
            astrocyte_1._d[comp].from_file(values_path)
    astrocyte_1.to_file(values_path)

    n_timesteps = len(next(iter(astrocyte_1._spike_comps.values())))
    time_start = astrocyte_1.time_start
    time_end = astrocyte_1.time_end
    t = np.linspace(time_start, time_end, n_timesteps)
    

    solution = astrocyte_1.solve() # computes the values over time
    diction = astrocyte_1._d # to look at the values for debuging
    y_axis_list = [r"[Ca$^{2+}$]$_c$ concentration in mM",r"[Ca$^{2+}$]$_{ER}$ concentration in mM",
                   r"[Ca$^{2+}$]$_{ES}$ concentration in mM",r"The fraction h of activated IP$_3$ receptor channels",
                   r"[IP$_3$]$_c$ concentration in mM",r"[Na$^+$]$_c$ concentration in mM",
                   r"[Na$^+$]$_{ES}$ concentration in mM",r"[K$^+$]$_c$ concentration in mM",
                   r"[K$^+$]$_{ES}$ concentration in mM",r"Membrane voltage in V"]
    
    y_min = np.inf
    y_max = -np.inf
    for i,comp in enumerate(astrocyte_1._d):
        y_plot = astrocyte_1.y_plot
        if y_min > min(solution.y[i*10+y_plot,:]):
            y_min = min(solution.y[i*10+y_plot,:])
        if y_max < max(solution.y[i*10+y_plot,:]):
            y_max = max(solution.y[i*10+y_plot,:])
    y_min -= (y_max-y_min)*.05
    y_max += (y_max-y_min)*.05
    ca_compartment_values = []
    freq = []
    compartment_curves_to_plot = astrocyte_model.compartment_time_curves_to_plot
    # print(astrocyte_1._d)

    for i,comp in enumerate(astrocyte_1._d):
        y_plot = astrocyte_1.y_plot
        ca_compartment_values.append(solution.y[i*10+y_plot,:])
        peaks, _ = sig.find_peaks(solution.y[i*10+y_plot,:], threshold=1e-7)
        freq.append(len(peaks)/41.66)
        if(comp in compartment_curves_to_plot):
            plt.figure(dpi=100)
            plt.title(comp)
            plt.plot(solution.t, solution.y[i*10+y_plot,:])
            plt.xlabel("time in s")
            plt.ylabel(y_axis_list[y_plot])
            plt.ylim([y_min,y_max])
            plt.grid()
            # plt.show()
            figname = 'cell'+str(astrocyte_model.cellID)+'_'+str(astrocyte_model.dataSet)+'_level'+str(astrocyte_model.maximum_level)+'_'+str(comp)+'.pdf'
            plt.savefig('output/'+figname)

    # print('length first process: ',len(processes_array[0]))  
    # print('length 7 process: ',len(processes_array[1])) 
    # print('length 8 process: ',len(processes_array[8]))    
    ca_compartment_values = np.round(np.max(ca_compartment_values[:],1),5)*1000
    freq = np.round(np.array(freq)/0.24,2)
    for i in processes_array:
        astrocyte_model.process_whole_list = astrocyte_model.process_whole_list + i
    astrocyte_model.ca_concentration_compartments = ca_compartment_values
    astrocyte_model.freq_compartments = freq
    # [svr, length, connection_matrix, processes_array] = astrocyte_model.loading_astrocyte_data(ca_compartment_plot=True)
    # [svr, length, connection_matrix, processes_array] = astrocyte_model.loading_astrocyte_data(freq_compartment_plot=True)

    save_Data_in_csv = True
    # if(save_Data_in_csv == True):
    #     for i,comp in enumerate(astrocyte_1._d):
    #         conc = np.concatenate(([solution.t], solution.y[i*10:(i+1)*10,:]), axis=0).transpose()
    #         df = pd.DataFrame(conc, columns = ["time[s]", "Ca_i", "Ca_ER", "Ca_es", "h", "IP3", "Na_i", "Na_es", "K_i", "K_es", "V_m"])
    #         df.to_csv(f"output/result_{comp}.csv", index=False)
    if(save_Data_in_csv == True):
        for i,comp in enumerate(astrocyte_1._d):
            if(comp in compartment_curves_to_plot):
                conc = np.concatenate(([solution.t], solution.y[i*10:(i+1)*10,:]), axis=0).transpose()
                df = pd.DataFrame(conc, columns = ["time[s]", "Ca_i", "Ca_ER", "Ca_es", "h", "IP3", "Na_i", "Na_es", "K_i", "K_es", "V_m"])
                df.to_csv(f"output/result_{comp}.csv", index=False)
    
    ende = time.time()
    ende_proc = time.process_time()
    print('Gesamtzeit: {:5.3f}s'.format(ende-start))
    print('Systemzeit: {:5.3f}s'.format(ende_proc-start_proc))