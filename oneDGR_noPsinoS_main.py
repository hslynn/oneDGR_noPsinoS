from __future__ import print_function
from dolfin import *
import matplotlib.pyplot as plt
import rk
import sys
import getopt
import time

from hdw import *
import bdry
import sch_kerr_schild_ingoing as sks
import mesh_generate as mg
import ioo

parameters["ghost_mode"] = "shared_vertex"

def main():
    """
    main computating process
    """

    N = 9
    DG_degree = 1
    inner_bdry = 0.5
    mesh_len = 1.0 
    hmin = 0.1 
    hmax = 0.5
    mg_func = 1
    mg_order = 1.0
    refine_time = 0
    folder = ''

    opts, dumps = getopt.getopt(sys.argv[1:], "-m:-d:-i:-h:-x:-r:-o:-l:-f:")
    for opt, arg in opts:
        if opt == "-m":
            mg_func = int(arg)
        if opt == "-d":
            DG_degree = int(arg)
        if opt == "-i":
            inner_bdry = float(arg)
        if opt == "-l":
            mesh_len = float(arg)
        if opt == "-h":
            hmin = float(arg)
        if opt == "-x":
            hmax = float(arg)
        if opt == "-o":
            mg_order = float(arg)
        if opt == "-r":
            refine_time = int(arg)
        if opt == "-f":
            folder = str(arg)
    
    #create .sh for rerun purpose
    with open(folder+'run.sh', 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('cd ~/projects/oneDGR_noPsinoS/\n')
        f.write('python3 oneDGR_noPsinoS_main.py ')
        f.write('-m %d -d %d -i %f -l %f -h %f -x %f -o %f -r %d -f %s'
                %(mg_func, DG_degree, inner_bdry, mesh_len, hmin, hmax, mg_order, refine_time, folder))

    #create mesh and define function space
    mesh = mg.get_mesh(inner_bdry, mesh_len, hmin, hmax, mg_func, mg_order)
    for dummy in range(refine_time):
        mesh = refine(mesh)
    print(mesh.hmin())
    print(mesh.hmax())
    print(mesh.num_vertices())
    print(mesh.coordinates())
    plot(mesh)
    plt.show()

    #save configurations to file
    with open(folder+'config.txt', 'w') as f:
        f.write("#configuratons used in this test\n\n")
        f.write("parameters to generate mesh:\n")
        f.write("inner_bdry = "+str(inner_bdry)+'\n')
        f.write("mesh_len = "+str(mesh_len)+'\n')
        f.write("mg_func = "+str(mg_func)+'\n')
        f.write("mg_order = "+str(mg_order)+'\n')
        f.write("hmin = "+str(hmin)+'\n')
        f.write("hmax = "+str(hmax)+'\n')
        f.write("refine_time = "+str(refine_time)+'\n\n')
        f.write("resulted mesh:\n")
        f.write("num_vertices = "+str(mesh.num_vertices())+'\n')
        f.write("minimum of h = "+str(mesh.hmin())+'\n')
        f.write("maximum of h = "+str(mesh.hmax())+'\n\n')
        f.write("others:\n")
        f.write("DG_degree = "+str(DG_degree))

    dt = hmin/(2*DG_degree + 1)
    func_space = FunctionSpace(mesh, "DG", DG_degree)
    func_space_accurate = FunctionSpace(mesh, "DG", DG_degree + 5)

    
    #coordinate function
    r = SpatialCoordinate(mesh)[0]

    #define functions for the variables
    var_list = [Function(func_space) for dummy in range(N)]
    var_list[0].rename('g00', 'g00')
    var_list[1].rename('g01', 'g01')
    var_list[2].rename('g11', 'g11')
    var_list[3].rename('Pi00', 'Pi00')
    var_list[4].rename('Pi01', 'Pi01')
    var_list[5].rename('Pi11', 'Pi11')
    var_list[6].rename('Phi00', 'Phi00')
    var_list[7].rename('Phi01', 'Phi01')
    var_list[8].rename('Phi11', 'Phi11')

    deri_list = [[Function(func_space), Function(func_space)] for dummy in range(len(var_list))]
    H_list = sks.get_H_list(func_space)
    deriH_list = sks.get_deriH_list(func_space)
    #define functions for the auxi variables
    invg_list = [Function(func_space) for dummy in range(3)]
    auxi_list = [Function(func_space) for dummy in range(5)]
    gamma_list = [Function(func_space) for dummy in range(8)]
    C_list = [Function(func_space) for dummy in range(2)]

    rhs_list = [Function(func_space) for dummy in range(N)]

    #create form for middle terms
    invg_forms = get_invg_forms(var_list)
    auxi_forms = get_auxi_forms(var_list, invg_list) 
    gamma_forms = get_gamma_forms(var_list, invg_list, auxi_list, r)
    C_forms = get_C_forms(H_list, gamma_list)

    src_forms = get_source_forms(var_list, invg_list, auxi_list, gamma_list, C_list, H_list, deriH_list, r)
    Hhat_forms = get_Hhat_forms(var_list, deri_list, auxi_list)
    rhs_forms = get_rhs_forms(Hhat_forms, src_forms)
    
    #pack forms and functions
    form_packs = (invg_forms, auxi_forms, gamma_forms, C_forms, rhs_forms) 
    func_packs = (invg_list, auxi_list, gamma_list, C_list, rhs_list)
    
    
    #Runge Kutta step
    #####################################################################################

    #initialize functions
    exact_var_list = sks.get_exact_var_list(func_space_accurate)
    project_functions(exact_var_list, var_list)
    temp_var_list = [Function(func_space) for dummy in range(N)]
    exact_characteristic_field_values = bdry.get_characteristic_field_values(exact_var_list)
   
    #functions used in diagnosting 
    Cr_forms = get_Cr_forms(var_list)
    Cr_list = [Function(func_space) for dummy in range(3)]
    dif_forms = [var_list[idx] - exact_var_list[idx] for idx in range(len(var_list))]
    dif_list = [Function(func_space) for dummy in range(len(var_list))]

    time_seq = []
    error_rhs_seqs = [[] for dummy in range(N)]
    error_var_seqs = [[] for dummy in range(N)]
    error_C_seqs = [[] for dummy in range(2)]
    error_Cr_seqs = [[] for dummy in range(3)]

    #read data from existing files if time_seq.txt exist, otherwise it means we start at t = 0.0
    try:
        with open(folder+'time_seq.txt', 'r') as f:
            time_seq = [float(t_point) for t_point in f.readlines()]
            t = time_seq[-1]
            ioo.read_var_from_files(var_list, folder)
            ioo.read_seqs_from_file(folder+'error_var_seqs.txt', error_var_seqs)
            ioo.read_seqs_from_file(folder+'error_rhs_seqs.txt', error_rhs_seqs)
            ioo.read_seqs_from_file(folder+'error_C_seqs.txt', error_C_seqs)
            ioo.read_seqs_from_file(folder+'error_Cr_seqs.txt', error_Cr_seqs)
    except FileNotFoundError:  
        t = 0.0

    t_end = 37.28444
    last_save_wall_time = time.time()
    plt.ion()
    fig_error = plt.figure(figsize=(19.2, 10.8))
    fig_C = plt.figure(figsize=(19.2, 10.8))
    fig_error_seq = plt.figure(figsize=(19.2, 10.8))
    fig_C_seq = plt.figure(figsize=(19.2, 10.8))
    fig_rhs_seq = plt.figure(figsize=(19.2, 10.8))
    while t < t_end:
        if t + dt < t_end:
            t += dt
        else:
            return '222'
            dt = t_end - t
            t = t_end
        time_seq.append(t)
        t_str = '%.05f'%t
        t_str = t_str.zfill(8)

        project_functions(var_list, temp_var_list)
        if DG_degree == 1:
            rk.rk2(var_list, exact_characteristic_field_values, temp_var_list, deri_list, form_packs, func_packs, dt)
        if DG_degree == 2:
            rk.rk3(var_list, exact_characteristic_field_values, temp_var_list, deri_list, form_packs, func_packs, dt)

        #print(find_AH(var_list, auxi_list, inner_bdry, inner_bdry+mesh_len, 0.001))
        
        project_functions(Cr_forms, Cr_list)
        project_functions(dif_forms, dif_list)
        for idx in range(len(var_list)):
            error_rhs = norm(rhs_list[idx], 'L2')
            error_var = errornorm(var_list[idx], exact_var_list[idx], 'L2')
            error_rhs_seqs[idx].append(error_rhs)
            error_var_seqs[idx].append(error_var)

        for idx in range(len(C_list)):
            error_C = norm(C_list[idx], 'L2')
            error_C_seqs[idx].append(error_C)

        for idx in range(len(Cr_list)):
            error_Cr = norm(Cr_list[idx], 'L2')
            error_Cr_seqs[idx].append(error_Cr)


        # show fig of real time error of vars and constraints, save both in .png form
        # constraint fig
        fig_C.clf()
        fig_C.suptitle('constraints when t = '+str(t))

        fig_C.add_subplot(2, 3, 1)
        plot_obj = fig_C.axes[0]
        ioo.plot_function(Cr_list[0], plot_obj)
        plot_obj.set_title('Cr00')

        fig_C.add_subplot(2, 3, 2)
        plot_obj = fig_C.axes[1]
        ioo.plot_function(Cr_list[1], plot_obj)
        plot_obj.set_title('Cr01')

        fig_C.add_subplot(2, 3, 3)
        plot_obj = fig_C.axes[2]
        ioo.plot_function(Cr_list[2], plot_obj)
        plot_obj.set_title('Cr11') 

        fig_C.add_subplot(2, 2, 3)
        plot_obj = fig_C.axes[3]
        ioo.plot_function(C_list[0], plot_obj)
        plot_obj.set_title('C0')

        fig_C.add_subplot(2, 2, 4)
        plot_obj = fig_C.axes[4]
        ioo.plot_function(C_list[1], plot_obj)
        plot_obj.set_title('C1')

        fig_C.savefig(folder+'constraint_'+t_str+'.png') 

        #error fig 
        fig_error.clf()
        fig_error.subplots(3, 3)
        fig_error.suptitle('error when t = '+str(t))
        
        for idx in range(9):
            plot_obj = fig_error.axes[idx]
            ioo.plot_function(dif_list[idx], plot_obj)
            plot_obj.set_title('error of '+str(var_list[idx].name()))

        fig_error.savefig(folder+'error_'+t_str+'.png') 
        plt.pause(0.000001)

        
        #save data and figs every 100 secends
        if time.time() - last_save_wall_time > 100.0 or t == t_end:
            for idx in range(len(var_list)): 
                var = var_list[idx]
                ufile = HDF5File(MPI.comm_world, folder+var.name()+".hdf5", 'w')
                ufile.write(var, var.name(), t) 
                ufile.close()
              
            ioo.write_seqs_to_file(folder+'error_var_seqs.txt', error_var_seqs)    
            ioo.write_seqs_to_file(folder+'error_rhs_seqs.txt', error_rhs_seqs)    
            ioo.write_seqs_to_file(folder+'error_C_seqs.txt', error_C_seqs)    
            ioo.write_seqs_to_file(folder+'error_Cr_seqs.txt', error_Cr_seqs)    
            with open(folder+'time_seq.txt', 'w') as f:
                for t_point in time_seq:
                    f.write(str(t_point)+'\n')

            last_save_wall_time = time.time()
            #save fig for error_var_seqs
            fig_error_seq.clf()
            fig_error_seq.suptitle('error over time')
            fig_error_seq.subplots(3, 3)

            for idx in range(9):
                plot_obj = fig_error_seq.axes[idx]
                plot_obj.plot(time_seq, error_var_seqs[idx], 'r') 
                plot_obj.set_title('error of '+str(var_list[idx].name())+' over time')
                plot_obj.set_xscale('log')
                plot_obj.set_yscale('log')
            
            fig_error_seq.savefig(folder+'error_over_time.png') 

            #rhs
            fig_rhs_seq.clf()
            fig_rhs_seq.suptitle('rhs over time')
            fig_rhs_seq.subplots(3, 3)

            for idx in range(9):
                plot_obj = fig_rhs_seq.axes[idx]
                plot_obj.plot(time_seq, error_rhs_seqs[idx], 'r') 
                plot_obj.set_xscale('log')
                plot_obj.set_yscale('log')
                plot_obj.set_title('L2 norm of rhs_'+str(var_list[idx].name())+' over time')

            fig_rhs_seq.savefig(folder+'rhs_over_time.png') 

            #constraint over time 
            fig_C_seq.clf()
            fig_C_seq.suptitle('constraint over time')

            fig_C_seq.add_subplot(2, 3, 1)
            plot_obj = fig_C_seq.axes[0]
            plot_obj.plot(time_seq, error_Cr_seqs[0], 'r') 
            plot_obj.set_xscale('log')
            plot_obj.set_yscale('log')
            plot_obj.set_title('L2 norm of Cr00 over time')

            fig_C_seq.add_subplot(2, 3, 2)
            plot_obj = fig_C_seq.axes[1]
            plot_obj.plot(time_seq, error_Cr_seqs[1], 'r') 
            plot_obj.set_xscale('log')
            plot_obj.set_yscale('log')
            plot_obj.set_title('L2 norm of Cr01 over time')
    
            fig_C_seq.add_subplot(2, 3, 3)
            plot_obj = fig_C_seq.axes[2]
            plot_obj.plot(time_seq, error_Cr_seqs[2], 'r') 
            plot_obj.set_xscale('log')
            plot_obj.set_yscale('log')
            plot_obj.set_title('L2 norm of Cr11 over time')
 
            fig_C_seq.add_subplot(2, 2, 3)
            plot_obj = fig_C_seq.axes[3]
            plot_obj.plot(time_seq, error_C_seqs[0], 'r') 
            plot_obj.set_xscale('log')
            plot_obj.set_yscale('log')
            plot_obj.set_title('L2 norm of C0 over time')

            fig_C_seq.add_subplot(2, 2, 4)
            plot_obj = fig_C_seq.axes[4]
            plot_obj.plot(time_seq, error_C_seqs[1], 'r') 
            plot_obj.set_xscale('log')
            plot_obj.set_yscale('log')
            plot_obj.set_title('L2 norm of C1 over time')
 
            fig_C_seq.savefig(folder+'constraint_over_time.png') 
 


    plt.ioff()
    print("\nMEOW!!")
main()
