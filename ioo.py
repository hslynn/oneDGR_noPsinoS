from dolfin import *
import numpy as np

def read_var_from_files(var_list, folder):
    for idx in range(len(var_list)):
        var = var_list[idx]
        ufile = HDF5File(MPI.comm_world, folder+var.name()+'.hdf5', 'r')
        ufile.read(var, var.name()) 
        ufile.close()

def write_seqs_to_file(fn, seqs):
    with open(fn, 'w') as f:
        for idx_time in range(len(seqs[0])):
            li = []
            for idx_obj in range(len(seqs)):
                li.append(str(seqs[idx_obj][idx_time])) 
            f.write(' '.join(li)+'\n')

def read_seqs_from_file(fn, seqs):
    with open(fn, 'r') as f:
        lines = f.readlines()
    for line in lines:
        values = line.split(" ")
        for idx in range(len(seqs)):
            seqs[idx].append(float(values[idx]))


def plot_function(func, plt_obj):
    mesh = func.function_space().mesh()
    num_cells = mesh.num_cells()
    for idx in range(num_cells):
        cell = Cell(mesh, idx)
        verts =  cell.get_vertex_coordinates()
        value_left = np.zeros(1, dtype=np.float64)
        value_right = np.zeros(1, dtype=np.float64)
        func.eval_cell(value_left, np.array([verts[0]]), cell)
        func.eval_cell(value_right, np.array([verts[1]]), cell)

        x_array = np.linspace(verts[0], verts[1], 2)
        y_array = np.array([func(x) for x in x_array])
        y_array[0] = value_left
        y_array[-1] = value_right
        plt_obj.plot(x_array, y_array, 'b')
       
    
