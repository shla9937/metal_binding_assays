#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.stats
import math
import sys


def usage():

    print("\nHere are the supported flags:\n")
    print("      -protein_conc <float in units of uM>")
    print("      -delta_Cp <float in units of kcal/mol.K>")
    print("      -list_of_concs <filename>")
    print("      -fluo_datafile <filename>")
    print("      -initial_parameters <filename>")
    print("      -output_parameters <filename>")
    print("      -skip_fitting")
    print("      -fit_with_shared_slopes")
    print("      -no_thermal_plots")
    print("      -make_separate_thermal_plots")
    print("      -no_Tm_shift_plots")
    print("      -no_isothermal_plots")
    print("      -isothermal_output_tag <tag>")
    print("      -temps_for_isothermal_plots <float1,float2,float3...>")
    print("      -version")
    print("\n\n")
    quit()
    
def main():

    if ( len(sys.argv) < 2 ):
        usage()

    # Fill in values for these parameters
    Cp = -1.0
    protein_conc = -1.0
    list_of_concs = ""
    fluo_datafile = ""
    initial_parameters = ""
    output_parameters = ""
    do_fitting = 1
    fit_with_shared_slopes = 0
    make_thermal_plots = 1
    separate_thermal_plots = 0
    make_Tm_plots = 1
    make_isothermal_plots = 1
    isothermal_output_tag = ""
    isothermal_Tlist = [];
    
    # parse command-line flags
    flags = sys.argv.copy()
    script_name = flags.pop(0)  # first item is the filename
    while ( len(flags) > 0 ):
        curr_flag = flags.pop(0)
        if ( curr_flag.lower() == "-help"):
            usage()
        elif ( curr_flag.lower() == "-usage"):
            usage()
        if ( curr_flag.lower() == "-version"):
            print("\nThis is version 1.0 of the DSF fitting code\n")
            quit()
        elif ( curr_flag.lower() == "-delta_cp"):
            if ( len(flags) == 0 ):
                print("Error: value for delta_Cp flag was not specified")
                quit()
            Cp = float(flags.pop(0))
            if ( Cp < 0 ):
                print("Error: Cp must be positive")
                quit()
        elif ( curr_flag.lower() == "-protein_conc"):
            if ( len(flags) == 0 ):
                print("Error: value for protein_conc flag was not specified")
                quit()
            protein_conc = float(flags.pop(0))
            if ( protein_conc < 0 ):
                print("Error: protein_conc must be positive")
                quit()
        elif ( curr_flag.lower() == "-list_of_concs"):
            if ( len(flags) == 0 ):
                print("Error: filename for list_of_concs flag was not specified")
                quit()
            list_of_concs = flags.pop(0)
            if ( list_of_concs.startswith('-') ):
                print("Error: the list_of_concs file was not specified (or starts with \-, which is strange)")
                quit()
        elif ( curr_flag.lower() == "-fluo_datafile"):
            if ( len(flags) == 0 ):
                print("Error: filename for fluo_datafile flag was not specified")
                quit()
            fluo_datafile = flags.pop(0)
            if ( fluo_datafile.startswith('-') ):
                print("Error: the fluo_datafile file was not specified (or starts with \-, which is strange)")
                quit()
        elif ( curr_flag.lower() == "-initial_parameters"):
            if ( len(flags) == 0 ):
                print("Error: filename for initial_parameters flag was not specified")
                quit()
            initial_parameters = flags.pop(0)
            if ( initial_parameters.startswith('-') ):
                print("Error: the initial_parameters file was not specified (or starts with \-, which is strange)")
                quit()
        elif ( curr_flag.lower() == "-output_parameters"):
            if ( len(flags) == 0 ):
                print("Error: filename for output_parameters flag was not specified")
                quit()
            output_parameters = flags.pop(0)
            if ( output_parameters.startswith('-') ):
                print("Error: the output_parameters file was not specified (or starts with \-, which is strange)")
                quit()
        elif ( curr_flag.lower() == "-skip_fitting"):
            do_fitting=0
        elif ( curr_flag.lower() == "-fit_with_shared_slopes"):
            fit_with_shared_slopes=1
        elif ( curr_flag.lower() == "-no_thermal_plots"):
            make_thermal_plots = 0
        elif ( curr_flag.lower() == "-make_separate_thermal_plots"):
            separate_thermal_plots = 1
        elif ( curr_flag.lower() == "-no_tm_shift_plots"):
            make_Tm_plots = 0
        elif ( curr_flag.lower() == "-no_isothermal_plots"):
            make_isothermal_plots = 0
        elif ( curr_flag.lower() == "-isothermal_output_tag"):
            if ( len(flags) == 0 ):
                print("Error: string for isothermal_output_tag flag was not specified")
                quit()
            isothermal_output_tag = flags.pop(0)
            if ( isothermal_output_tag.startswith('-') ):
                print("Error: the isothermal_output_tag string was not specified (or starts with \-, which is strange)")
                quit()
        elif ( curr_flag.lower() == "-temps_for_isothermal_plots"):
            if ( len(flags) == 0 ):
                print("Error: temperatures for isothermal plots were not specified")
                quit()
            temps_for_isothermal_plots = flags.pop(0)
            if ( temps_for_isothermal_plots.startswith('-') ):
                print("Error: the temperatures for isothermal plots were not specified (or starts with \-, which is strange)")
                quit()
            isothermal_Tlist = temps_for_isothermal_plots.split(',')
        else:
            print("\nError: unrecognized flag", curr_flag)
            print("\nFor a list of recognized flags, type:    ",script_name,"-help\n")
            quit()

    if ( Cp < 0 ):
        print("Error: must specify a value for Cp")
        quit()
    if ( protein_conc < 0 ):
        print("Error: must specify a value for protein_conc")
        quit()
    if ( list_of_concs == "" ):
        print("Error: must specify a file containing the list of ligand concentrations")
        quit()
    if ( fluo_datafile == "" ):
        print("Error: must specify a file containing the fluorescence data")
        quit()

    if ( len(isothermal_Tlist) == 0 ):
        make_isothermal_plots = 0
    elif ( isothermal_output_tag == "" ):
        print("Error: must specify isothermal_output_tag for isothermal plots")
        quit()
        

    # Read the ligand concentrations and fluorescence data from separate files
    # Note: there's a 1-line header expected on the fluorescence data file
    print("Reading input data")
    concs = np.loadtxt(list_of_concs)
    all_dat = np.loadtxt(fluo_datafile, skiprows=1, delimiter="\t")

    # Temperatures are in the first column, make these a separate array
    temperatures = all_dat[:,0]
    all_dat = np.delete(all_dat, 0, 1)
    
    if (len(concs) != len(all_dat[0,:])):
        print("Mismatch in concs and all_dat array lengths")
        quit()
        
    print("Proceeding with {} temperatures".format(len(temperatures)))
    print("Proceeding with {} ligand concentrations".format(len(concs)))

    fitting_params = np.empty([6,len(concs)])
    if ( initial_parameters != "" ):
        print("Reading initial parameters")
        fitting_params = np.loadtxt(initial_parameters, skiprows=1).transpose()
    else:
        print("Estimating initial parameters from the data")
        fitting_params = estimate_initial_params(concs, temperatures, all_dat, Cp, fit_with_shared_slopes)

    if ( do_fitting == 1 ):
        if ( fit_with_shared_slopes ):
            print("Carrying out a global fit of the thermal unfolding curves (shared slopes)")
#            print("Note: if previous fits were with fixed slope, those will be initial guesses now")
#            print("Note: if not, only some concentrations for estimating initial slopes at this time")
            mean_unfolded_slope = np.mean(fitting_params[4,0:2])
            final_pt = len(concs)-1
            mean_folded_slope = np.mean(fitting_params[5,final_pt-2:final_pt])
            fitting_params = fit_global_thermal_curves(len(concs), fitting_params, temperatures, all_dat, mean_unfolded_slope, mean_folded_slope, Cp)

        else:
            print("Fitting thermal curves individually")
            fitting_params = fit_all_individual_thermal_curves(concs, temperatures, all_dat, Cp)
    else:
        print("Proceeding with input (or estimated) parameters, no fitting will be carried out")

    if ( output_parameters != "" ):
        np.savetxt(output_parameters, fitting_params.transpose(), fmt='%15.5f', header="        Tm (oC)     dH (kcal/mol)       intercept_U       intercept_F           slope_U           slope_F", delimiter="   ", comments="")


    # Make thermal unfolding plots
    plot_thermal_curve_fits=1
    if ( make_thermal_plots ):
        plot_fits_to_thermal_unfolding_data(concs, temperatures, fitting_params, all_dat, Cp)

    # Make Tm-shift plots
    if ( make_Tm_plots ):
        plot_Tm_shifts(concs, fitting_params[0,:])

    unique_concs = np.unique(concs)
    all_isothermal_data = np.empty([len(concs),len(isothermal_Tlist)])
    all_isothermal_fits = np.empty([len(unique_concs),len(isothermal_Tlist)])
    for t_index in range(len(isothermal_Tlist)):
        binding_temperature = float(isothermal_Tlist[t_index])
        fraction_unfolded = calculate_fraction_unfolded( binding_temperature, concs, fitting_params, Cp )
        all_isothermal_data[:,t_index] = fraction_unfolded.transpose()
        ( Ku, Kd, fit_values ) = fit_isothermal_Kd( fraction_unfolded, concs, unique_concs, protein_conc )
        all_isothermal_fits[:,t_index] = fit_values.transpose()

        print('At temperature %5.2f oC, Ku is %4.4f and Kd is %4.4f uM.' % (binding_temperature, Ku, Kd))
        if ( isothermal_output_tag != "" ):
            fname_isothermal_data = isothermal_output_tag+"_T="+isothermal_Tlist[t_index]+".data.txt"
            fname_isothermal_fit = isothermal_output_tag+"_T="+isothermal_Tlist[t_index]+".fit.txt"
            fd = open(fname_isothermal_data, 'w')
            ff = open(fname_isothermal_fit, 'w')
            fd.write("    conc (uM)\tfraction_unfolded\n")
            ff.write("    conc (uM)\tfraction_unfolded\n")
            for c_index in range(len(concs)):
                fd.write('{:13.5f}'.format(concs[c_index])+"\t"+'{:17.5f}'.format(fraction_unfolded[c_index])+"\n")
            for c_index in range(len(unique_concs)):
                ff.write('{:13.5f}'.format(unique_concs[c_index])+"\t"+'{:17.5f}'.format(fit_values[c_index])+"\n")
            fd.close()
            ff.close()

    # Make isothermal plots
    if ( make_isothermal_plots ):
        plot_isothermal_binding_fits(isothermal_Tlist, concs, all_isothermal_data, unique_concs, all_isothermal_fits)
            
    plt.show()
    print("Done fitting program")

    
# Error function for initial first-pass fitting of thermal unfolding curves
def single_thermal_curve_errfxn( x0, fluorescence, temperatures, Cp ):

    # Unpack x0
    if ( len(x0) != 6 ):
        print("Error passing x0 to single_thermal_curve_errfxn")
        quit()
    Tm = x0[0]
    dH = x0[1]
    unfolded_intercept = x0[2]
    folded_intercept = x0[3]
    unfolded_slope = x0[4]
    folded_slope = x0[5]

    if (len(fluorescence) != len(temperatures)):
        print("Mismatch in fluorescence and T array lengths")
        quit()

    sq_err = 0
    for index in range(len(temperatures)):
        T = temperatures[index] + 273.15
        R = 1.987/1000
        dG = dH*(1-T/(Tm+273.15)) - Cp*(Tm+273.15-T + T*np.log(T/(Tm+273.15)))
        Ku = np.exp(-dG/(R*T))
        Y = (Ku/(1+Ku))*(unfolded_slope*T + unfolded_intercept) + (1/(1+Ku))*(folded_slope*T + folded_intercept)
        err = Y - fluorescence[index]
        sq_err += err*err

    return sq_err

# Error function for initial first-pass fitting of thermal unfolding curves
def single_thermal_curve_errfxn_fixed_slopes( x0, fluorescence, temperatures, unfolded_slope, folded_slope, Cp ):

    # Unpack x0
    if ( len(x0) != 4 ):
        print("Error passing x0 to single_thermal_curve_errfxn_fixed_slopes")
        quit()
    Tm = x0[0]
    dH = x0[1]
    unfolded_intercept = x0[2]
    folded_intercept = x0[3]

    if (len(fluorescence) != len(temperatures)):
        print("Mismatch in fluorescence and T array lengths")
        quit()

    sq_err = 0
    for index in range(len(temperatures)):
        T = temperatures[index] + 273.15
        R = 1.987/1000
        dG = dH*(1-T/(Tm+273.15)) - Cp*(Tm+273.15-T + T*np.log(T/(Tm+273.15)))
        Ku = np.exp(-dG/(R*T))
        Y = (Ku/(1+Ku))*(unfolded_slope*T + unfolded_intercept) + (1/(1+Ku))*(folded_slope*T + folded_intercept)
        err = Y - fluorescence[index]
        sq_err += err*err

    return sq_err

# Carry out initial first-pass fitting of thermal unfolding curves
def fit_single_thermal_curve(temperatures, fluorescence, Cp):
    
    init_dH = 150
    window_size = int(len(temperatures)/10)
    ( init_bottom_slope, init_bottom_intercept, junk1, junk2, junk3 ) = scipy.stats.linregress(temperatures[0:window_size]+273.15, fluorescence[0:window_size])
    tse=len(temperatures)-1
    ( init_top_slope, init_top_intercept, junk1, junk2, junk3 ) = scipy.stats.linregress(temperatures[tse-window_size:tse]+273.15, fluorescence[tse-window_size:tse])
    fmin, fmax = np.amin(fluorescence), np.amax(fluorescence)
    fluo_midpoint = fmin + (fmax - fmin) / 2
    diff = abs(fluo_midpoint*10)
    init_Tm = 0
    for index in range(len(temperatures)):
        curr_diff = abs( fluorescence[index] - fluo_midpoint)
        if (curr_diff < diff) :
            init_Tm = temperatures[index]
            diff = curr_diff

    # Pack x0
    x0 = np.empty([6])
    x0[0] = init_Tm
    x0[1] = init_dH
    x0[2] = init_top_intercept
    x0[3] = init_bottom_intercept
    x0[4] = init_top_slope
    x0[5] = init_bottom_slope
            
# Powell is faster, but Nelder-Mead seems to give much better fits
#    res = scipy.optimize.minimize( single_thermal_curve_errfxn, x0, args=(fluorescence, temperatures, Cp), method='Powell' )
    res = scipy.optimize.minimize( single_thermal_curve_errfxn, x0, args=(fluorescence, temperatures, Cp), method='Nelder-Mead' )

#    print(res)
#    min_Tm = res.x[0]
#    min_dH = res.x[1]
#    min_ti = res.x[2]
#    min_bi = res.x[3]
#    min_ts = res.x[4]
#    min_bs = res.x[5]
#    print("Tm is",min_Tm,"dH is",min_dH)
    
    return res.x

# Carry out initial first-pass fitting of thermal unfolding curves
def fit_single_thermal_curve_fixed_slopes(temperatures, fluorescence, unfolded_slope, folded_slope, Cp):
    
    # start with a standard estimate of dH
    init_dH = 150
    # estimate the unfolded intercept using the (fixed) unfolded slope
    last_inx = len(temperatures)-1
    init_top_intercept = fluorescence[last_inx] - unfolded_slope * (temperatures[last_inx]+273.15)
    # estimate the folded intercept using the (fixed) folded slope
    init_bottom_intercept = fluorescence[0] - folded_slope * (temperatures[0]+273.15)

    # estimate the Tm based on the midpoint of the curve
    fluo_midpoint = ( np.amax(fluorescence) - np.amin(fluorescence) ) / 2
    diff = abs(fluo_midpoint*10)
    init_Tm = 0
    for index in range(len(temperatures)):
        curr_diff = abs( fluorescence[index] - fluo_midpoint)
        if (curr_diff < diff) :
            init_Tm = temperatures[index]
            diff = curr_diff

    # Pack x0
    x0 = np.empty([4])
    x0[0] = init_Tm
    x0[1] = init_dH
    x0[2] = init_top_intercept
    x0[3] = init_bottom_intercept
            
# Powell is faster, but Nelder-Mead seems to give much better fits
#    res = scipy.optimize.minimize( single_thermal_curve_errfxn_fixed_slopes, x0, args=(fluorescence, temperatures, unfolded_slope, folded_slope, Cp), method='Powell' )
    res = scipy.optimize.minimize( single_thermal_curve_errfxn_fixed_slopes, x0, args=(fluorescence, temperatures, unfolded_slope, folded_slope, Cp), method='Nelder-Mead' )

#    print(res)
#    min_Tm = res.x[0]
#    min_dH = res.x[1]
#    min_ti = res.x[2]
#    min_bi = res.x[3]
#    print("Tm is",min_Tm,"dH is",min_dH)
    
    return res.x




def fit_all_individual_thermal_curves( concs, temperatures, all_dat, Cp):

    num_datasets = len(concs)
    single_fit_params = np.empty([6,num_datasets])
    for index in range(num_datasets):
        fit_params = fit_single_thermal_curve(temperatures,all_dat[:,index],Cp)
#        single_fit_params[:,index] = fit_params
        single_fit_params[0,index] = fit_params[0]
        single_fit_params[1,index] = fit_params[1]
        single_fit_params[2,index] = fit_params[2]
        single_fit_params[3,index] = fit_params[3]
        single_fit_params[4,index] = fit_params[4]
        single_fit_params[5,index] = fit_params[5]
        min_Tm = fit_params[0]
        min_dH = fit_params[1]
#        min_ti = fit_params[2]
#        min_bi = fit_params[3]
#        min_ts = fit_params[4]
#        min_bs = fit_params[5]
#        print("For ligand conc %.3f uM, Tm is %.2f oC and delta_H is %.2f kcal/mol." % (concs[index], min_Tm, min_dH))
        print('For ligand conc %.3f uM, Tm is %.2f oC.' % (concs[index], min_Tm))

    return single_fit_params


def fit_all_individual_thermal_curves_fixed_slopes( concs, temperatures, all_dat, unfolded_slope, folded_slope, Cp):

    num_datasets = len(concs)
    single_fit_params = np.empty([6,num_datasets])

    for index in range(num_datasets):
        fit_params = fit_single_thermal_curve_fixed_slopes(temperatures,all_dat[:,index], unfolded_slope, folded_slope, Cp)
#        single_fit_params[:,index] = fit_params
        single_fit_params[0,index] = fit_params[0]
        single_fit_params[1,index] = fit_params[1]
        single_fit_params[2,index] = fit_params[2]
        single_fit_params[3,index] = fit_params[3]
        single_fit_params[4,index] = unfolded_slope
        single_fit_params[5,index] = folded_slope
        min_Tm = fit_params[0]
        min_dH = fit_params[1]
#        min_ti = fit_params[2]
#        min_bi = fit_params[3]
#        print("For ligand conc %.3f uM, Tm is %.2f oC and delta_H is %.2f kcal/mol." % (concs[index], min_Tm, min_dH))
        print('For ligand conc %.3f uM, Tm is %.2f oC.' % (concs[index], min_Tm))
        
    return single_fit_params



# Error function for global fitting of thermal unfolding curves
def global_thermal_curve_errfxn( curr_param_values, all_dat, temperatures, Cp ):

    num_datasets = int((len(curr_param_values) - 2 ) / 4)
    unfolded_slope = curr_param_values[0]
    folded_slope = curr_param_values[1]

    sq_err = 0
    for conc_index in range(num_datasets):
            Tm = curr_param_values[4*conc_index+2]
            dH = curr_param_values[4*conc_index+3]
            unfolded_intercept = curr_param_values[4*conc_index+4]
            folded_intercept = curr_param_values[4*conc_index+5]
#            print("Inside global fitting, Tm is", Tm)
            for t_index in range(len(temperatures)):
                T = temperatures[t_index] + 273.15
                R = 1.987/1000
                dG = dH*(1-T/(Tm+273.15)) - Cp*(Tm+273.15-T + T*np.log(T/(Tm+273.15)))
                Ku = np.exp(-dG/(R*T))
                Y = (Ku/(1+Ku))*(unfolded_slope*T + unfolded_intercept) + (1/(1+Ku))*(folded_slope*T + folded_intercept)
                err = Y - all_dat[t_index, conc_index]
                sq_err += err*err

    return sq_err

# Run global (shared) fitting of thermal unfolding curves
def fit_global_thermal_curves(num_datasets, single_fit_params, temperatures, all_dat, shared_unfolded_slope, shared_folded_slope, Cp):

#    print("init shared unfolded slope is", shared_unfolded_slope, "and folded slope is", shared_folded_slope)

    # Shared free params in this fit are going to be folded_slope and unfolded_slope and dH, *separate* params are going to be the folded/unfolded intercepts, and the Tm's
    init_params = np.empty([4*num_datasets+2])
    init_params[0] = shared_unfolded_slope
    init_params[1] = shared_folded_slope
    last_temp = len(temperatures)-1
    for index in range(num_datasets):
        init_params[4*index+2] = single_fit_params[0,index]   # Tm from earlier fit
        init_params[4*index+3] = single_fit_params[1,index]   # dH from earlier fit
        # update the estimated intercepts now that we're using different (shared) slopes
        init_params[4*index+4] = all_dat[last_temp,index] - shared_unfolded_slope * (temperatures[last_temp]+273.15) # unfolded intercept
        init_params[4*index+5] = all_dat[0,index] - shared_folded_slope * (temperatures[0]+273.15) # folded intercept

#    print("init_params are", init_params)
        
    # Powell is faster, but Nelder-Mead seems to give much better fits
#    res = scipy.optimize.minimize( global_thermal_curve_errfxn, init_params, args=(all_dat, temperatures, Cp), method='Powell' )
    res = scipy.optimize.minimize( global_thermal_curve_errfxn, init_params, args=(all_dat, temperatures, Cp), method='Nelder-Mead' )
#    print(res)

    global_fit_params = np.copy(single_fit_params)
    for index in range(num_datasets):
        global_fit_params[0,index] = res.x[4*index+2]   # Tm
        global_fit_params[1,index] = res.x[4*index+3]   # dH
        global_fit_params[2,index] = res.x[4*index+4]   # unfolded intercept
        global_fit_params[3,index] = res.x[4*index+5]   # folded intercept
        global_fit_params[4,index] = res.x[0]   # (shared) unfolded slope
        global_fit_params[5,index] = res.x[1]   # (shared) folded slope

    return global_fit_params


# Build the (isothermal) fraction folded curve at a specified temperature
def calculate_fraction_unfolded(binding_temperature, concs, fitting_params, Cp):
    
    num_datasets = len(concs)
    fraction_unfolded = np.empty((concs).shape)
    for c_index in range(num_datasets):
        Tm = fitting_params[0,c_index]   # Tm
        dH = fitting_params[1,c_index]   # dH
        #ti = fitting_params[2,c_index]   # unfolded intercept
        #bi = fitting_params[3,c_index]   # folded intercept
        #ts = fitting_params[4,c_index]   # unfolded slope
        #bs = fitting_params[5,c_index]   # folded slope

        T = binding_temperature + 273.15
        R = 1.987/1000
        dG = dH*(1-T/(Tm+273.15)) - Cp*(Tm+273.15-T + T*np.log(T/(Tm+273.15)))
        Ku = np.exp(-dG/(R*T))
        fraction_unfolded[c_index] = (Ku/(1+Ku))
        
    return fraction_unfolded


# Use Ku and Kd (and protein+ligand concs) to get the fraction unfolded
def calculate_fitted_isothermal_point(ligand_conc, Ku, Kd, protein_conc):
    Kd = abs(Kd)
    Ku = abs(Ku)
    b = protein_conc + Kd*(1+Ku) - ligand_conc
    c = -1.0 * ligand_conc * Kd * (1+Ku)
    L_free = (-b+math.sqrt(b*b-4*c))/2
    # assume that L_free = L_tot (if we don't want to use the quadratic equation above)
    #L_free = concs[index]
    # then use L_free to get fraction unfolded
    fit_fraction_unfolded = 1 / (1 + (1/Ku)*(1+L_free/Kd))
    return fit_fraction_unfolded


            
# Fitting function for binding curves
def binding_curve_errfxn( x0, fraction_unfolded, concs, protein_conc ):

    # Unpack x0
    if ( len(x0) != 2 ):
        print("Error passing x0 to binding_curve_errfxn")
        quit()
    Ku = x0[0]
    Kd = x0[1]
    
    if (len(fraction_unfolded) != len(concs)):
        print("Mismatch in fraction_unfolded and concs array lengths")
        quit()

    sq_err = 0
    for index in range(len(concs)):
        fit_fraction_unfolded = calculate_fitted_isothermal_point(concs[index], Ku, Kd, protein_conc)
        err = fit_fraction_unfolded - fraction_unfolded[index]
        sq_err += err*err
    return sq_err


# Fit the (isothermal) binding curves
def fit_isothermal_Kd( fraction_unfolded, concs, unique_concs, protein_conc ):

    if (len(fraction_unfolded) != len(concs)):
        print("Mismatch in fraction_unfolded and concs array lengths")
        quit()

    max_Fu = 0
    for index in range(len(fraction_unfolded)):
        if ( fraction_unfolded[index] > max_Fu ):
            max_Fu = fraction_unfolded[index]
    init_Ku = max_Fu/(1.0-max_Fu)
    init_midpoint = max_Fu / 2
    init_EC50 = 0
    diff = 1
    for index in range(len(fraction_unfolded)):
        curr_diff = abs( fraction_unfolded[index] - init_midpoint)
        if (curr_diff < diff) :
            init_EC50 = concs[index]
            diff = curr_diff
    init_Kd = max_Fu * ( init_EC50 - protein_conc / 2 )
    
    # Pack x0
    x0 = np.empty([2])
    x0[0] = init_Ku
    x0[1] = init_Kd

    # Powell is faster, but Nelder-Mead seems to give much better fits
#    res = scipy.optimize.minimize( binding_curve_errfxn, x0, args=(fraction_unfolded, concs, protein_conc), method='Powell' )
    res = scipy.optimize.minimize( binding_curve_errfxn, x0, args=(fraction_unfolded, concs, protein_conc), method='Nelder-Mead' )

#    print(res)
    Ku = abs(res.x[0])
    Kd = abs(res.x[1])
#    print("Kd is",Kd,"and Ku is", Ku)

    fit_values = np.empty((unique_concs).shape)
    for c_index in range(len(unique_concs)):
        fit_values[c_index] = calculate_fitted_isothermal_point(unique_concs[c_index], Ku, Kd, protein_conc)

    return ( Ku, Kd, fit_values )


    
# Make plots of the fits compared to the experimental thermal unfolding curves
def plot_fits_to_thermal_unfolding_data(concs, temperatures, fitting_params, all_dat, Cp ):
    
    num_datasets = len(concs)
    # figure out good dimensions for the subplot step (in case we're using it)
    subplot_horiz = math.ceil(math.sqrt(num_datasets))
    subplot_vert = math.ceil(num_datasets / subplot_horiz)

    plt.figure(5, figsize=(18,11))

    for c_index in range(num_datasets):
        Tm = fitting_params[0,c_index]   # Tm
        dH = fitting_params[1,c_index]   # dH
        ti = fitting_params[2,c_index]   # unfolded intercept
        bi = fitting_params[3,c_index]   # folded intercept
        ts = fitting_params[4,c_index]   # unfolded slope
        bs = fitting_params[5,c_index]   # folded slope

        ax = plt.subplot(subplot_vert, subplot_horiz, c_index+1)

        fluorescence = all_dat[:,c_index]
        fit_fluo = np.empty((fluorescence).shape)
        for t_index in range(len(temperatures)):
            T = temperatures[t_index] + 273.15
            R = 1.987/1000
            dG = dH*(1-T/(Tm+273.15)) - Cp*(Tm+273.15-T + T*np.log(T/(Tm+273.15)))
            Ku = np.exp(-dG/(R*T))
            fit_fluo[t_index] = (Ku/(1+Ku))*(ts*T + ti) + (1/(1+Ku))*(bs*T + bi)
    
        plt.plot(temperatures, fluorescence, 'bs', temperatures, fit_fluo, 'r')

        if ( (c_index % subplot_horiz) == 0 ):
            # print the y-axis label only on the left-most column
            plt.ylabel('Fluorescence')
        # take the numbers off all columns (not just the middle ones)
        ax.set_yticklabels([])
        if ( ( c_index + subplot_horiz >= num_datasets ) ):
            # print the x-axis label only on the bottom row
            plt.xlabel('Temperature (oC)')
        else:
            # take the numbers off all other rows
            ax.set_xticklabels([])
        plt.title('Ligand conc. '+str(concs[c_index])+' uM')
            
    # make the plot
    plt.savefig('thermal_unfolding_curves.png', dpi=100)
    plt.show(block=False)

    return 0


# Make plots of the fits compared to the experimental thermal unfolding curves
def plot_Tm_shifts(concs, Tms):
    # Plot the Tm shifts
    plt.figure(4, figsize=(8,6))
    plt.semilogx(concs, Tms, 'ro')
    plt.xlabel('Ligand concentration (uM)')
    plt.ylabel('Tm (oC)')
    plt.title('Tm-shifts')
    plt.savefig('Tm_shifts.png', dpi=100)
    plt.show(block=False)

    
# Make plots of the fits compared to the isothermal fraction folded values
def plot_isothermal_binding_fits(isothermal_Tlist, concs, all_isothermal_data, unique_concs, all_isothermal_fits ):
    
    num_plots = len(isothermal_Tlist)
    # figure out good dimensions for the subplot step (in case we're using it)
    subplot_horiz = math.ceil(math.sqrt(num_plots))
    subplot_vert = math.ceil(num_plots / subplot_horiz)

    plt.figure(3, figsize=(14,12))

    for t_index in range(len(isothermal_Tlist)):
        temperature_string = isothermal_Tlist[t_index]                
        ax = plt.subplot(subplot_vert, subplot_horiz, t_index+1)
        plt.semilogx(concs, all_isothermal_data[:,t_index], 'bs', unique_concs, all_isothermal_fits[:,t_index], 'r')

        if ( (t_index % subplot_horiz) == 0 ):
            # print the y-axis label only on the left-most column
            plt.ylabel('Fraction unfolded')
        if ( ( t_index + subplot_horiz >= num_plots ) ):
            # print the x-axis label only on the bottom row
            plt.xlabel('Ligand concentration (uM)')
#        else:
            # take the numbers off all other rows
#            ax.set_xticklabels([])
        plt.title('Isothermal data at '+temperature_string+' oC')
            
    # make the plot (with many panels)
    plt.savefig('isothermal_curves.single.png', dpi=100)
    plt.show(block=False)

    # make another plot with all the curves on a single plot
    if ( num_plots > 1 ):
        plt.figure(2, figsize=(8,6))
        for t_index in range(len(isothermal_Tlist)):
            temperature_string = isothermal_Tlist[t_index]                
            plt.semilogx(concs, all_isothermal_data[:,t_index], 'bs', unique_concs, all_isothermal_fits[:,t_index], 'r')
            plt.xlabel('Ligand concentration (uM)')
            plt.ylabel('Fraction unfolded')
            plt.title('Isothermal data at all requested temperatures')
            plt.savefig('isothermal_curves.all.png', dpi=100)
            plt.show(block=False)
    
    return 0


def estimate_initial_params( concs, temperatures, all_dat, Cp, shared_slopes ):
    
    num_datasets = len(concs)
    init_params = np.empty([6,num_datasets])
    init_dH = 150
    window_size = int(len(temperatures)/10)

    init_unfolded_slope=0
    init_folded_slope=0
    init_unfolded_intercept=0
    init_folded_intercept=0

    # For now, we require that the FIRST concentrations are the lowest ligand concentrations, and the LAST concentrations are the highest.
    # JK For later, we could remove this requirement. But for now, just test to make sure the first/last values match the lowest/highest
    if ( np.amin(concs) != concs[0] ):
        print("Error, the first concentration is not the lowest")
        quit()
    if ( np.amax(concs) != concs[num_datasets-1] ):
        print("Error, the last concentration is not the highest")
        quit()

    if ( shared_slopes ):
        # estimate the unfolded slope as the average of the FIRST three concentrations / replicates (assuming these are the lowest ligand concentrations)
        tse=len(temperatures)-1
        ( init_unfolded_slopeA, init_unfolded_interceptA, junk1, junk2, junk3 ) = scipy.stats.linregress(temperatures[tse-window_size:tse]+273.15, all_dat[tse-window_size:tse,0])
        ( init_unfolded_slopeB, init_unfolded_interceptB, junk1, junk2, junk3 ) = scipy.stats.linregress(temperatures[tse-window_size:tse]+273.15, all_dat[tse-window_size:tse,1])
        ( init_unfolded_slopeC, init_unfolded_interceptC, junk1, junk2, junk3 ) = scipy.stats.linregress(temperatures[tse-window_size:tse]+273.15, all_dat[tse-window_size:tse,2])
        init_unfolded_slope = ( init_unfolded_slopeA + init_unfolded_slopeB + init_unfolded_slopeC ) / 3.0
        # estimate the folded slope as the average of the LAST three concentrations / replicates (assuming these are the highest ligand concentrations)
        ( init_folded_slopeA, init_folded_interceptA, junk1, junk2, junk3 ) = scipy.stats.linregress(temperatures[0:window_size]+273.15, all_dat[0:window_size,num_datasets-3])
        ( init_folded_slopeB, init_folded_interceptB, junk1, junk2, junk3 ) = scipy.stats.linregress(temperatures[0:window_size]+273.15, all_dat[0:window_size,num_datasets-2])
        ( init_folded_slopeC, init_folded_interceptC, junk1, junk2, junk3 ) = scipy.stats.linregress(temperatures[0:window_size]+273.15, all_dat[0:window_size,num_datasets-1])
        init_folded_slope = ( init_folded_slopeA + init_folded_slopeB + init_folded_slopeC ) / 3.0

    for c_index in range(num_datasets):
        fluorescence = all_dat[:,c_index]
        use_shared_slope_estimates = 1   # even though we won't fix or share the slopes, start fitting from shared values (recommended, but it probably doesn't matter)
        if ( use_shared_slope_estimates ):
            # estimate just the intercepts (from the single point at the highest/lowest temperature), since we're keeping the shared slopes from above
            last_inx = len(temperatures)-1
            init_unfolded_intercept = fluorescence[last_inx] - init_unfolded_slope * (temperatures[last_inx]+273.15)
            init_bottom_intercept = fluorescence[0] - init_folded_slope * (temperatures[0]+273.15)
        else:
            # estimate both slope and intercept from each dataset
            ( init_folded_slope, init_folded_intercept, junk1, junk2, junk3 ) = scipy.stats.linregress(temperatures[0:window_size]+273.15, fluorescence[0:window_size])
            tse=len(temperatures)-1
            ( init_unfolded_slope, init_unfolded_intercept, junk1, junk2, junk3 ) = scipy.stats.linregress(temperatures[tse-window_size:tse]+273.15, fluorescence[tse-window_size:tse])

        fluo_midpoint = ( np.amax(fluorescence) - np.amin(fluorescence) ) / 2
        diff = abs(fluo_midpoint*10)
        init_Tm = 0
        for t_index in range(len(temperatures)):
            curr_diff = abs( fluorescence[t_index] - fluo_midpoint)
            if (curr_diff < diff) :
                init_Tm = temperatures[t_index]
                diff = curr_diff
        init_params[0,c_index] = init_Tm
        init_params[1,c_index] = init_dH
        init_params[2,c_index] = init_unfolded_intercept
        init_params[3,c_index] = init_folded_intercept
        init_params[4,c_index] = init_unfolded_slope
        init_params[5,c_index] = init_folded_slope

    return init_params


if __name__ == '__main__':
    main()
