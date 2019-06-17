from __future__ import print_function
from dolfin import *
import matplotlib.pyplot as plt
import rk
import sys
import getopt


from hdw import *
import sch_kerr_schild_ingoing as sks
import mesh_generate as mg
import read_var_from_files as rvff

parameters["ghost_mode"] = "shared_vertex"

def main():
    """
    main computating process
    """
    N = 9
    DG_degree = 2
    mesh_num = 10
    inner_bdry = 0.5
    mesh_len = 3.0 
    hmin = 0.1 
    mg_order = 1.0

    opts, dumps = getopt.getopt(sys.argv[1:], "-m:-d:-i:-h:-o:")
    for opt, arg in opts:
        if opt == "-m":
            mesh_num = int(arg)
        if opt == "-d":
            DG_degree = int(arg)
        if opt == "-i":
            inner_bdry = float(arg)
        if opt == "-h":
            hmin = float(arg)
        if opt == "-o":
            mg_order = float(arg)


    #create mesh and define function space
    #mesh = IntervalMesh(mesh_num, inner_bdry, inner_bdry + mesh_len)
    mesh = mg.get_mesh(inner_bdry, mesh_len, hmin, mg_order)
    h = mesh.hmin()
    print(h)
    func_space = FunctionSpace(mesh, "DG", DG_degree)
    dt = h/(2*DG_degree + 1)

    print(mesh.num_vertices())
    plot(mesh)
    plt.show()

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

    Hhat_list = [Function(func_space) for dummy in range(N)]
    src_list = [Function(func_space) for dummy in range(N)]
    rhs_list = [Function(func_space) for dummy in range(N)]

    #create form for middle terms
    invg_forms = get_invg_forms(var_list)
    auxi_forms = get_auxi_forms(var_list, invg_list) 
    gamma_forms = get_gamma_forms(var_list, invg_list, auxi_list, r)
    C_forms = get_C_forms(H_list, gamma_list)

    Hhat_forms = get_Hhat_forms(var_list, deri_list, auxi_list)
    src_forms = get_source_forms(var_list, invg_list, gamma_list, auxi_list, C_list, H_list, deriH_list, r)
    rhs_forms = get_rhs_forms(Hhat_forms, src_forms)
    
    #pack forms and functions
    form_packs = (invg_forms, auxi_forms, gamma_forms, C_forms, Hhat_forms, src_forms, rhs_forms) 
    func_packs = (invg_list, auxi_list, gamma_list, C_list, Hhat_list, src_list, rhs_list)
    
    #Runge Kutta step
    exact_var_list = sks.get_exact_var_list(func_space)
    project_functions(exact_var_list, var_list)
    temp_var_list = [Function(func_space) for dummy in range(N)]
    characteristic_field_values = get_characteristic_field_values(exact_var_list)

    t_now = 0.0
    t_end = 100.0
    time_seq = []
    error_rhs_seq = [[] for dummy in range(N)]
    error_var_seq = [[] for dummy in range(N)]
    error_C0_seq = []
    error_C1_seq = []
    zero_func = project(Expression("0.", degree=10), func_space)

    dif_forms = [var_list[idx] - exact_var_list[idx] for idx in range(N)]
    dif_list = [Function(func_space) for dummy in range(N)]
    plt.ion()


    with open('time_savepoint.txt', 'r') as f:
        lines = f.readlines()
        last_save_time = float(lines[-1])

    t_now = last_save_time
    rvff.read_var_from_files(var_list)
    
    while t_now + dt <= t_end:
        project_functions(var_list, temp_var_list)
        rk.rk3(exact_var_list, var_list, characteristic_field_values, temp_var_list, deri_list, form_packs, func_packs, dt)

        #print(find_AH(var_list, auxi_list, inner_bdry, inner_bdry+mesh_len, 0.001))

        error_rhs = [errornorm(rhs, zero_func, 'L2') for rhs in rhs_list]
        error_C0 = errornorm(C_list[0], zero_func, 'L2')
        error_C1 = errornorm(C_list[1], zero_func, 'L2')
        error_var = [errornorm(var_list[idx], exact_var_list[idx], 'L2') for idx in range(len(var_list))]
        error_C0_seq.append(error_C0)
        error_C1_seq.append(error_C1)
        for idx in range(len(var_list)):
            error_rhs_seq[idx].append(error_rhs[idx])
            error_var_seq[idx].append(error_var[idx])

        t_now += dt
        time_seq.append(t_now)
        
        if t_now - last_save_time > 0.2:
            for idx in range(len(var_list)): 
                var = var_list[idx]
                ufile = HDF5File(MPI.comm_world, var.name()+".hdf5", 'w')
                ufile.write(var, var.name(), t_now) 
                ufile.close()
                
            timefile = open('time_savepoint.txt', 'a')
            timefile.write(str(t_now)+"\n")
            timefile.close()
   
            last_save_time = t_now

        plt.clf()
        
        project_functions(dif_forms, dif_list)
        plt.subplot(4, 4, 1)
        plot(dif_list[0]) 
        plt.title('error of g00')
        
        plt.subplot(4, 4, 2)
        plot(dif_list[1])
        plt.title('error of g01')

        plt.subplot(4, 4, 3)
        plot(dif_list[2])
        plt.title('error of g11')

        plt.subplot(4, 4, 4)
        plt.plot(time_seq, error_var_seq[0], 'r')
        plt.title('L2 error of g00 evolved in time')


        plt.subplot(4, 4, 5)
        plot(dif_list[6])
        plt.title('error of Phi00')

        plt.subplot(4, 4, 6)
        plot(dif_list[7])
        plt.title('error of Phi01')

        plt.subplot(4, 4, 7)
        plot(dif_list[8])
        plt.title('error of Phi11')

        plt.subplot(4, 4, 8)
        plt.plot(time_seq, error_var_seq[1], 'r')
        plt.title('L2 error of g01 evolved in time')


        plt.subplot(4, 4, 9)
        plot(dif_list[3])
        plt.title('error of Pi00')

        plt.subplot(4, 4, 10)
        plot(dif_list[4])
        plt.title('error of Pi01')

        plt.subplot(4, 4, 11)
        plot(dif_list[5])
        plt.title('error of Pi11')

        plt.subplot(4, 4, 12)
        plt.plot(time_seq, error_var_seq[2], 'r')
        plt.title('L2 error of g11 evolved in time')


        plt.subplot(4, 4, 13)
        plot(C_list[0])
        plt.title('C0')

        plt.subplot(4, 4, 14)
        plot(C_list[1])
        plt.title('C1')

        plt.subplot(4, 4, 15)
        plt.plot(time_seq, error_C0_seq, 'r')
        plt.title('L2 norm of constraint C0')

        plt.subplot(4, 4, 16)
        plt.plot(time_seq, error_C1_seq, 'r')
        plt.title('L2 norm of constraint C1')

        plt.savefig(str(t_now)+'_var.png') 
        plt.pause(0.000001)
        

    plt.pause(100000000)
    if t_now < t_end:
        project_functions(var_list, temp_var_list)
        rk.rk3(exact_var_list, var_list, incoming_field_values, temp_var_list, deri_list, form_packs, func_packs, t_end-t_now)

    plt.pause(100000)
    plt.ioff()

main()
