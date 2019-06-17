"""
generating mesh
"""

from dolfin import *

def get_mesh(inner_bdry, mesh_len, hmin, hmax, mg_func, order):
    out_bdry = inner_bdry + mesh_len

    vertex = out_bdry
    vertex_list = []
    while vertex >= inner_bdry:
        vertex_list.append(vertex)
        if mg_func == 1:
            if abs((vertex-2.0)*(inner_bdry-vertex)) < 5:
                vertex -= min(hmax, order**(abs((vertex-2.0)*(inner_bdry-vertex)))*hmin) #mg_func = 1
            else:
                vertex -= hmax
        if mg_func == 2:
            vertex -= min(hmax, order**(vertex - inner_bdry)*hmin) #mg_func=2
        
    if vertex_list[-1] > inner_bdry:
        #vertex_list.append(inner_bdry)
        vertex_list.append(vertex)
    vertex_list.reverse()

    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, 'interval', 1, 1)
    editor.init_vertices(len(vertex_list))
    editor.init_cells(len(vertex_list)-1)
    for idx in range(len(vertex_list)):
        editor.add_vertex(idx, Point(vertex_list[idx]))
        if idx > 0: 
            editor.add_cell(idx - 1, [idx - 1, idx])
    editor.close()
    return mesh
