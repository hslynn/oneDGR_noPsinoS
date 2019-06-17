from dolfin import *
import hdw
import bdry

def rk2(var_list, exact_characteristic_field_values, temp_var_list, deri_list, form_packs, func_packs, dt):
    auxi_list = func_packs[1]
    rhs_forms = form_packs[-1]
    rhs_list = func_packs[-1]

    for dummy in range(2): 
        for idx in range(len(form_packs)-1):
            hdw.project_functions(form_packs[idx], func_packs[idx])

        bdry_values = bdry.get_bdry_values(var_list, auxi_list, exact_characteristic_field_values)
        for idx in range(len(var_list)):
            hdw.get_deri(deri_list[idx][0], var_list[idx], bdry_values[idx], 0, "+")
            hdw.get_deri(deri_list[idx][1], var_list[idx], bdry_values[idx], 0, "-")
        
        hdw.project_functions(rhs_forms, rhs_list)
        dt_forms = [var_list[idx] + dt*rhs_list[idx] for idx in range(len(var_list))]
        hdw.project_functions(dt_forms, var_list)

    final_forms = [0.5*(temp_var_list[idx] + var_list[idx]) for idx in range(len(var_list))]
    hdw.project_functions(final_forms, var_list) 


def rk3(var_list, exact_characteristic_field_values, temp_var_list, deri_list, form_packs, func_packs, dt):
    auxi_list = func_packs[1]
    rhs_forms = form_packs[-1]
    rhs_list = func_packs[-1]
    
    #compute u1, stored in var_list
    for idx in range(len(form_packs)-1):
        hdw.project_functions(form_packs[idx], func_packs[idx])

    bdry_values = bdry.get_bdry_values(var_list, auxi_list, exact_characteristic_field_values)
    for idx in range(len(var_list)):
        hdw.get_deri(deri_list[idx][0], var_list[idx], bdry_values[idx], 0, "+")
        hdw.get_deri(deri_list[idx][1], var_list[idx], bdry_values[idx], 0, "-")

    hdw.project_functions(rhs_forms, rhs_list)
    u1_forms = [temp_var_list[idx] + dt*rhs_list[idx] for idx in range(len(var_list))]
    hdw.project_functions(u1_forms, var_list)

    #compute u2, stored in var_list
    for idx in range(len(form_packs)-1):
        hdw.project_functions(form_packs[idx], func_packs[idx])
    bdry_values = bdry.get_bdry_values(var_list, auxi_list, exact_characteristic_field_values)
    for idx in range(len(var_list)):
        hdw.get_deri(deri_list[idx][0], var_list[idx], bdry_values[idx], 0, "+")
        hdw.get_deri(deri_list[idx][1], var_list[idx], bdry_values[idx], 0, "-")

    hdw.project_functions(rhs_forms, rhs_list)
    u2_forms = [3.0/4.0*temp_var_list[idx] + 1.0/4.0*var_list[idx] + 1.0/4.0*dt*rhs_list[idx] for idx in range(len(var_list))]
    hdw.project_functions(u2_forms, var_list)

    #compute final u, stored in var_list
    for idx in range(len(form_packs)-1):
        hdw.project_functions(form_packs[idx], func_packs[idx])

    bdry_values = bdry.get_bdry_values(var_list, auxi_list, exact_characteristic_field_values)
    for idx in range(len(var_list)):
        hdw.get_deri(deri_list[idx][0], var_list[idx], bdry_values[idx], 0, "+")
        hdw.get_deri(deri_list[idx][1], var_list[idx], bdry_values[idx], 0, "-")
    hdw.project_functions(rhs_forms, rhs_list)
    final_forms = [1.0/3.0*temp_var_list[idx] + 2.0/3.0*var_list[idx] + 2.0/3.0*dt*rhs_list[idx] for idx in range(len(var_list))]
    hdw.project_functions(final_forms, var_list) 
