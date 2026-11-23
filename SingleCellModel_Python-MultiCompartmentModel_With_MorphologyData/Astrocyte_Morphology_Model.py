"""
Created on Nov 2022

@author: rene
"""
from configparser import ConfigParser
from utils import config_writer
from utils.writable import Writable
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.linalg import null_space
 
    
class AstrocyteMorphologyModel(Writable):

    #init values
    _name = "AstrocyteMorphologyModel"
    cellID = 45
    dataSet = 'H00'
    oneBranch = False
    maximum_level = 80
    mainBranch = 8
    smallest_diameter=0
    radius_soma = 'diame_max'
    compartment_time_curves_to_plot = ['compartment_0']
    
    process_whole_list = []
    ca_concentration_compartments = []
    freq_compartments = []

    def get_name(self):
        return self._name
    
    def cylinder_plot(self, fig, ax, point_1_pos,point_2_pos,
                      compartment_diameter,compartment_color):
        pointsnumber=20
        r = compartment_diameter/2
        datapoints = np.linspace(0,2*np.pi,pointsnumber)
        X = np.zeros([2,pointsnumber])
        Y = np.zeros([2,pointsnumber])
        Z = np.zeros([2,pointsnumber])
        z_vector = (point_2_pos-point_1_pos)/np.linalg.norm(point_2_pos-point_1_pos)
        orth_space = null_space(z_vector.reshape([1,3]))
        x_vector = (orth_space[:,0]/np.linalg.norm(orth_space[:,0])).T
        y_vector = (orth_space[:,1]/np.linalg.norm(orth_space[:,1])).T
        
        X[0,:] = point_1_pos[0]+r*np.cos(datapoints)*x_vector[0]+  \
                 r*np.sin(datapoints)*y_vector[0]
        Y[0,:] = point_1_pos[1]+r*np.cos(datapoints)*x_vector[1]+  \
                 r*np.sin(datapoints)*y_vector[1]
        Z[0,:] = point_1_pos[2]+r*np.cos(datapoints)*x_vector[2]+  \
                 r*np.sin(datapoints)*y_vector[2]
        
        X[1,:] = point_1_pos[0]+(point_2_pos[0]-point_1_pos[0])+  \
                 r*np.cos(datapoints)*x_vector[0]+r*np.sin(datapoints)*y_vector[0]
        Y[1,:] = point_1_pos[1]+(point_2_pos[1]-point_1_pos[1])+  \
                 r*np.cos(datapoints)*x_vector[1]+r*np.sin(datapoints)*y_vector[1]
        Z[1,:] = point_1_pos[2]+(point_2_pos[2]-point_1_pos[2])+  \
                 r*np.cos(datapoints)*x_vector[2]+r*np.sin(datapoints)*y_vector[2]
        
        X_circle1 = np.zeros([2,pointsnumber])
        Y_circle1 = np.zeros([2,pointsnumber])
        Z_circle1 = np.zeros([2,pointsnumber])
        X_circle1[0,:] = np.zeros(len(X_circle1[0,:]))+point_1_pos[0] 
        Y_circle1[0,:] = np.zeros(len(X_circle1[0,:]))+point_1_pos[1] 
        Z_circle1[0,:] = np.zeros(len(X_circle1[0,:]))+point_1_pos[2]
        X_circle1[1,:] = point_1_pos[0]+r*np.cos(datapoints)*x_vector[0]+  \
                         r*np.sin(datapoints)*y_vector[0] 
        Y_circle1[1,:] = point_1_pos[1]+r*np.cos(datapoints)*x_vector[1]+  \
                         r*np.sin(datapoints)*y_vector[1]
        Z_circle1[1,:] = point_1_pos[2]+r*np.cos(datapoints)*x_vector[2]+  \
                         r*np.sin(datapoints)*y_vector[2]
        
        X_circle2 = np.zeros([2,pointsnumber])
        Y_circle2 = np.zeros([2,pointsnumber])
        Z_circle2 = np.zeros([2,pointsnumber])
        X_circle2[0,:] = np.zeros(len(X_circle1[0,:]))+point_2_pos[0] 
        Y_circle2[0,:] = np.zeros(len(X_circle1[0,:]))+point_2_pos[1] 
        Z_circle2[0,:] = np.zeros(len(X_circle1[0,:]))+point_2_pos[2]
        X_circle2[1,:] = point_2_pos[0]+r*np.cos(datapoints)*x_vector[0]+  \
                         r*np.sin(datapoints)*y_vector[0]
        Y_circle2[1,:] = point_2_pos[1]+r*np.cos(datapoints)*x_vector[1]+  \
                         r*np.sin(datapoints)*y_vector[1]
        Z_circle2[1,:] = point_2_pos[2]+r*np.cos(datapoints)*x_vector[2]+  \
                         r*np.sin(datapoints)*y_vector[2]
        
        cmap_col = plt.cm.get_cmap('spring')
        compartment_color = (np.squeeze(cmap_col(compartment_color)))
    
        ax.plot_surface(X,Y,Z,color=compartment_color)
        ax.plot_surface(X_circle1,Y_circle1,Z_circle1,color=compartment_color)
        ax.plot_surface(X_circle2,Y_circle2,Z_circle2,color=compartment_color)
    
    
    def loading_astrocyte_data(self,
                               ca_compartment_plot=False,
                               freq_compartment_plot=False):
        #load the Exel data
        positionData = pd.read_csv('positionData.csv')
        diameterData = pd.read_csv('diameterData.csv')
        #branchData is of type mat, therefore loadmat
        branchmat = loadmat('branchData.mat')
        branchData = branchmat['branchData']
        #give the branchData a header so that the entries can be called by them
        header = ['MatlabData','ExcelData','Depth', 'Level', 'Index']
        branchData = pd.DataFrame(branchData, columns=header)
        
        #extract position and diameter data for specific ID from whole dataset
        filamentID = positionData.FilamentID[self.cellID-1]
        diameterDataCell = diameterData[diameterData.FilamentID == filamentID]
        positionDataCell = positionData[positionData.FilamentID == filamentID]
        #reset the index of the data for a correct indexing
        diameterDataCell.reset_index(inplace=True, drop=True)
        positionDataCell.reset_index(inplace=True, drop=True)
        
        maximum_depth = max(diameterDataCell.Depth)
        maximum_diameter = max(diameterDataCell.PtDiameter)
        
        soma_pos = np.array([positionDataCell.PtPositionX[0],
                             positionDataCell.PtPositionY[0],
                             positionDataCell.PtPositionZ[0]])
        
        #calculates radius of soma from data with depth 1
        posData_Depth_1 = positionDataCell[positionDataCell.Depth ==1]
        posData_Depth_1.reset_index(inplace=True, drop=True)
        distance = np.zeros(len(posData_Depth_1.Depth))
        for i in range(0,len(posData_Depth_1.Depth)):
            point_depth_1 = np.array([posData_Depth_1.PtPositionX[i],
                                      posData_Depth_1.PtPositionY[i],
                                      posData_Depth_1.PtPositionZ[i]])
            distance[i]= np.linalg.norm(point_depth_1-soma_pos)   
        r_soma = 0
        if (self.radius_soma == 'depth_min'):
            r_soma = np.min(distance)
        elif (self.radius_soma == 'depth_max'):
            r_soma = np.max(distance)
        elif (self.radius_soma == 'diame_max'):
            r_soma = np.max(diameterDataCell.PtDiameter)/2
            
        number_of_compartments = 0 #just to know it but is not used
        
        '''return values: svr, length and connection_matrix. The variables
        contain all existing branch compartments and later they are reduced 
        to the appropiate size for the given maximum level'''
        svr = np.zeros(positionDataCell.shape[0])
        length = np.zeros(positionDataCell.shape[0])
        connection_matrix = np.zeros([positionDataCell.shape[0]+1,
                                      positionDataCell.shape[0]+1],dtype=int) #*2 von anna
        connection_dict = {} #anna
        '''For each compartment which is constructed, a new list entry is saved 
        in the right position for constructing the connection matrices later'''
        process_list = []
        #everytime a new main branch is generated this variable is incremented
        new_list_pos = -1 
        #list position in process_list for the actual branch in the loop
        list_pos = 0 
        
        #Setup the plot
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection ='3d')

        if self.maximum_level == "all":
            self.maximum_level=(np.max(branchData['Level'])[0,0])
        #loop over the levels
        for current_level in range(1,self.maximum_level+1):

            #takes branch data for that specific level
            levelDataCell = branchData[branchData['Level'] == current_level] 
            levelDataCell.reset_index(inplace=True, drop=True)
            #used to get the first index of the point which is really ploted
            process_index_begin = 0
            #loop over points in this level to create compartments for each connection
            for current_branch_connection in range(0,len(levelDataCell['Level'])):
                index1 = np.squeeze(levelDataCell['ExcelData']
                                    [current_branch_connection])[0]-1
                index2 = np.squeeze(levelDataCell['ExcelData']
                                    [current_branch_connection])[1]-1
                #get point ID, needed for getting diameters at those points
                point_id1 = positionDataCell.ID[index1]
                point_id2 = positionDataCell.ID[index2]
                #get diameters at these 2 points
                dia_1 = np.array(diameterDataCell[
                        diameterDataCell.ID == point_id1].PtDiameter)
                dia_2 = np.array(diameterDataCell[
                        diameterDataCell.ID == point_id2].PtDiameter)
                #get exact position for these 2 points
                point_1_pos = np.array([positionDataCell.PtPositionX[index1],
                                        positionDataCell.PtPositionY[index1],
                                        positionDataCell.PtPositionZ[index1]])
                point_2_pos = np.array([positionDataCell.PtPositionX[index2],
                                        positionDataCell.PtPositionY[index2],
                                        positionDataCell.PtPositionZ[index2]])
                #calculate mean diameter
                compartment_diameter = (dia_1+dia_2)/2
                #color of the compartment
                compartment_color = compartment_diameter/maximum_diameter
                #calculate point distance to soma
                soma_distance_point_2_pos = np.linalg.norm(point_2_pos-soma_pos)
                soma_distance_point_1_pos = np.linalg.norm(point_1_pos-soma_pos)
                
                #only plot compartment if it is not in soma
                if soma_distance_point_2_pos>r_soma: 
                    #in case point one is in soma, shift it outside
                    if soma_distance_point_1_pos<r_soma: 
                        point_1_pos = point_1_pos+(point_2_pos-point_1_pos)/np.linalg.norm(
                                     point_2_pos-point_1_pos)*(r_soma-soma_distance_point_1_pos)
                    current_depth = np.array(diameterDataCell[
                                    diameterDataCell.ID == point_id2].Depth)
                    '''Is the current depth smaller than the maximum depth and 
                    the diameter greater than the smallest allowed one?'''
                    process_list_copy = list(np.copy(process_list))
                    if len(process_list_copy)==0:
                        process_list_copy.append(['not_index1'])

                    if self.oneBranch: #choosing if compartment gets added depending whether whole cell or single branch should get plotted
                        ifCondition = (current_depth <= maximum_depth) and (compartment_diameter >= self.smallest_diameter) and (current_level==self.mainBranch or index1 in process_list_copy[0]) #anna(last condition)
                    else:
                        ifCondition = (current_depth <= maximum_depth) and (compartment_diameter >= self.smallest_diameter)# and (current_level==self.mainBranch or index1 in process_list_copy[0])

                    if ifCondition:
                        number_of_compartments += 1
                        if len(connection_dict)==0:
                            connection_dict[str(index1)]=1
                        elif str(index1) not in connection_dict:
                            connection_dict[str(index1)]=max(connection_dict.values())+1 # number_of_compartments*2-1
                        if str(index2) not in connection_dict:
                            connection_dict[str(index2)]=max(connection_dict.values())+1 #number_of_compartments*2
                        # print(index1,index2)
                        # print(connection_dict)
                        index1_helper=np.copy(connection_dict[str(index1)])
                        index2_helper=np.copy(connection_dict[str(index2)])
                        # print(index1_helper,index2_helper)
                        #set the connection in the connection matrix 
                        if current_branch_connection == process_index_begin:
                            #branch connected to soma?
                            connection_matrix[0,index2_helper] = 1
                            connection_matrix[index2_helper,0] = 1
                        else:
                            connection_matrix[index1_helper,index2_helper] = 1
                            connection_matrix[index2_helper,index1_helper] = 1
                        
                        '''determine the process_list position or if it is a 
                        new main branch, create a new list and insert it'''  
                        if current_branch_connection == process_index_begin:
                        #only done once for the first "to-plot" point in the branch    
                            append_new_list = True
                            if index1 == 0:
                                append_new_list = True
                            else:
                                for index_process_list in range(0,len(process_list)):
                                    #control if branch is a side branch
                                    if index1 in process_list[index_process_list]:
                                        list_pos = index_process_list
                                        append_new_list = False
                                        break
                            if(append_new_list == True):
                                new_list_pos += 1
                                list_pos = new_list_pos
                                process_list.append([])
                        
                        #insert the index in the right process_list position
                        process_list[list_pos].append(index2)
                        #calculate the svr at that index
                        svr[index2] = 4/(compartment_diameter*1e-6)
                        #calculate the length at that index
                        length[index2] = np.linalg.norm(point_1_pos-point_2_pos)*1e-6
                        
                        #include this compartment to the plot by the cylinder_plot()
                        if ca_compartment_plot:
                            self.cylinder_plot(fig, ax, point_1_pos,point_2_pos,
                                               compartment_diameter,
                                               np.array([self.ca_concentration_compartments[self.process_whole_list.index(index2)]]))
                        elif freq_compartment_plot:
                            self.cylinder_plot(fig, ax, point_1_pos,point_2_pos,
                                               compartment_diameter,
                                               np.array([self.freq_compartments[self.process_whole_list.index(index2)]]))
                        else:
                            self.cylinder_plot(fig, ax, point_1_pos,point_2_pos,
                                               compartment_diameter,compartment_color)
                else:
                #first compartment lies completely in the soma, use the next one
                    process_index_begin += 1 
        # print(connection_dict)
        print('Number of compartments: ',number_of_compartments)
        #the soma is a sphere
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
        x = r_soma*np.cos(u)*np.sin(v)
        y = r_soma*np.sin(u)*np.sin(v)
        z = r_soma*np.cos(v)
        
        #Color map which is used
        cmap_col = plt.cm.get_cmap('spring')
        
        #Genertate the specific kind of plot
        if ca_compartment_plot:
            ax.plot_surface(x+soma_pos[0], y+soma_pos[1], z+soma_pos[2], 
                            color=cmap_col(self.ca_concentration_compartments[0]))
            color_bar_norm = plt.Normalize(vmin=0, vmax=1)
            sm = plt.cm.ScalarMappable(cmap=cmap_col, norm=color_bar_norm)
            plt.colorbar(sm, label="Ca$^{2+}$ concentration in $\mu$M")
        elif freq_compartment_plot:
            ax.plot_surface(x+soma_pos[0], y+soma_pos[1], z+soma_pos[2], 
                            color=cmap_col(self.freq_compartments[0]))
            color_bar_norm = plt.Normalize(vmin=0, vmax=0.24)
            sm = plt.cm.ScalarMappable(cmap=cmap_col, norm=color_bar_norm)
            plt.colorbar(sm, label="frequency in Hz")
        else:
            compartment_color = (np.squeeze(cmap_col((2*r_soma)/maximum_diameter)))
            ax.plot_surface(x+soma_pos[0], y+soma_pos[1], z+soma_pos[2], 
                            color=compartment_color)
            color_bar_norm = plt.Normalize(vmin=0, vmax=maximum_diameter*1e-6)
            sm = plt.cm.ScalarMappable(cmap=cmap_col, norm=color_bar_norm)
            plt.colorbar(sm, label="compartment diameter in m",ax=plt.gca())
        
        #adjust the view point and set the plot title
        ax.view_init(elev=10, azim=-100)
        plt.title('cellID ' + str(self.cellID) + ' depth:' + str(maximum_depth) + 
                  ' levels:' + str(self.maximum_level), fontsize=20)    
        figname = 'cell'+str(self.cellID) +'_'+str(self.dataSet)+'_level'+str(self.maximum_level)+'_morphology.pdf'
        plt.savefig('output/'+figname)
        # plt.show()
        
        '''reset the connections to soma, because there will be a connection per 
        definition for each first sublist entry of process_list''' 
        connection_matrix[0,:] = 0
        connection_matrix[:,0] = 0
        
        #generate return lists with first entries for the soma
        connection_matrix_list = [[1]]
        svr_list = [3/(r_soma*1e-6)]
        length_list = [2*r_soma*1e-6]
        
        '''now we add the elements from the numpy arrays to the return lists
        by looping over the sublists/main branches in the process_list'''
        for j in range(0,len(process_list)):
            help_matrix = connection_matrix.copy()
            #add all elements of this sublist to the svr and length list
            svr_list = svr_list + svr[process_list[j]].tolist()
            length_list = length_list + length[process_list[j]].tolist()
            
            #elements which should be not in the connection_matrix for this branch
            deletion_list = [] 
            for list_index in range(0,len(process_list)):
                if(list_index == j):
                    continue
                else:
                    deletion_list = deletion_list + process_list[list_index]
            for i in range(0,len(deletion_list)):
                help_matrix[deletion_list[i],:] = 0
                help_matrix[:,deletion_list[i]] = 0
            #squezze the actual connection matrix so that it has the right form
            help_matrix = help_matrix[~np.all(help_matrix == 0, axis=1)]
            help_matrix = (help_matrix.T[~np.all(help_matrix.T == 0, axis=1)]).T
            help_matrix += np.eye(help_matrix.shape[0], dtype=int)
            #is there only a connection to the soma -> empty matrix
            if(np.any(help_matrix)==False):
                help_matrix = np.array([1])
            #add connection matrix for this main branch to the whole connection list
            connection_matrix_list.append(help_matrix.tolist())
        
        return svr_list, length_list, connection_matrix_list, process_list
        
        
    def to_file(self, path):
        config_parser = ConfigParser()
        config_writer.write_recursively(self, config_parser)

        with open(path, 'w') as f:
            config_parser.write(f)

    def from_file(self, path):
        config_parser = ConfigParser(inline_comment_prefixes="#")
        config_parser.read(path)
        config_writer.read_recursively(self, config_parser)
