#!/usr/bin/env python
#
#    This file is part of iSpec.
#    Copyright Sergi Blanco-Cuaresma - http://www.blancocuaresma.com/s/
#
#    iSpec is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    iSpec is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with iSpec. If not, see <http://www.gnu.org/licenses/>.
#
import os
import sys
import numpy as np
import logging
import multiprocessing
from multiprocessing import Pool

################################################################################
#--- iSpec directory -------------------------------------------------------------
objName = sys.argv[1]
if len(sys.argv) > 2:
    free_param = sys.argv[2]
else:
    free_param = 9
ispec_dir = os.path.dirname(os.path.realpath(__file__)) + "/"
mySamIn_dir = "mySample/input/%7s/" % objName
mySamOut_dir = "mySample/output/%7s/" % objName
if not os.path.exists(mySamOut_dir):
    os.makedirs(mySamOut_dir)
sys.path.insert(0, os.path.abspath(ispec_dir))

#--- Change LOG level ----------------------------------------------------------
#LOG_LEVEL = "warning"
LOG_LEVEL = "info"
logger = logging.getLogger() # root logger, common for all
logger.setLevel(logging.getLevelName(LOG_LEVEL.upper()))
################################################################################
#--- Average stellar parameter and variables definition
initial_teff = 7554.0; initial_logg = 1.04; initial_MH = -0.92 #J003643
#initial_teff = 7394.0; initial_logg = 0.81; initial_MH = -1.05
#initial_teff = 4900.0; initial_logg = 0.0; initial_MH = -1.5
#initial_teff = 4500.0; initial_logg = 1.0; initial_MH = -1.5
initial_vmic = 4.10; initial_R = 45000.
star_spectrum = []; star_continuum_model = []; star_continuum_regions= []
estimated_snr = []; segments = []; star_linemasks = []



def LinUpd(): #To update the value of free parameter
    text = ""
    f = open(ispec_dir + 'ispec/lines.py', 'r+')
    for line in f:
        if "free_param = " in line:
            lineX="    free_param = "+str(free_param)+"\n"
            line = line.replace(line,lineX)
        text += line
    f.seek(0)
    f.write(text)
    f.truncate()
    f.close()
  
def ContFitAndNorm(star_spectrum):
    global initial_R
    logging.info("Fitting continuum...")
    model = "Splines" # "Polynomy"
    degree = 2
    from_resolution = initial_R
    
    # Strategy: Filter first median values and secondly MAXIMUMs in order to find the continuum
    order='median+max'
    logging.info("MEDIAN WAVE RANGE =" + str(np.median(star_spectrum['waveobs'][1:]-star_spectrum['waveobs'][:-1])))
    median_wave_range=3.*np.median(star_spectrum['waveobs'][1:]-star_spectrum['waveobs'][:-1])
    max_wave_range=30.*np.median(star_spectrum['waveobs'][1:]-star_spectrum['waveobs'][:-1])
    nknots = int((np.max(star_spectrum['waveobs']) - np.min(star_spectrum['waveobs'])) / max_wave_range)+1 #len(segments) # Automatic: 1 spline every 5 nm
    
    star_continuum_model = ispec.fit_continuum(star_spectrum, from_resolution=initial_R, \
                                nknots=nknots, degree=degree, \
                                median_wave_range=median_wave_range, \
                                max_wave_range=max_wave_range, \
                                model=model, order=order, \
                                automatic_strong_line_detection=True, \
                                strong_line_probability=0.5, \
                                use_errors_for_fitting=True)
    #--- Normalize ----------------------------------------------------------
    star_spectrum = ispec.normalize_spectrum(star_spectrum, star_continuum_model, consider_continuum_errors=False)
    star_continuum_model = ispec.fit_continuum(star_spectrum, fixed_value=1.0, model="Fixed value")
    return (star_spectrum, star_continuum_model)
   
def CosmicFilter(spec, star_continuum_model):
    # Spectrum should be already normalized
    blue_spec = spec[spec['waveobs'] <= 465.]
    green_spec = spec[(spec['waveobs'] >= 465.) & (spec['waveobs'] <= 580.)]
    red_spec = spec[spec['waveobs'] >= 580.]
    blue_step = blue_spec['waveobs'][len(blue_spec)//2+1] - blue_spec['waveobs'][len(blue_spec)//2]
    green_step = green_spec['waveobs'][len(green_spec)//2+1] - green_spec['waveobs'][len(green_spec)//2]
    red_step = red_spec['waveobs'][len(red_spec)//2+1] - red_spec['waveobs'][len(red_spec)//2]
    cosmics1 = ispec.create_filter_cosmic_rays(blue_spec,star_continuum_model,\
              resampling_wave_step=blue_step, window_size=15, \
              variation_limit=0.01)
    blue_step = blue_spec[~cosmics1]
    cosmics2 = ispec.create_filter_cosmic_rays(green_spec,star_continuum_model,\
              resampling_wave_step=green_step, window_size=15, \
              variation_limit=0.01)
    green_step = green_spec[~cosmics2]
    cosmics3 = ispec.create_filter_cosmic_rays(red_spec,star_continuum_model,\
              resampling_wave_step=red_step, window_size=15, \
              variation_limit=0.01)
    red_step = red_spec[~cosmics3]
    star_spectrum = np.concatenate((blue_spec, green_spec, red_spec), axis=0)
    return(star_spectrum)

def ContReg(star_spectrum, chip):
    global initial_R
    logging.info("Finding continuum regions...")
    sigma = 0.025#0.002
    max_continuum_diff = 0.05#0.01
    star_continuum_regions = ispec.find_continuum(star_spectrum, \
            resolution=initial_R, max_std_continuum = sigma, \
            continuum_model = 0.95, \
            max_continuum_diff=max_continuum_diff, \
            fixed_wave_step=None)
    ispec.write_continuum_regions(star_continuum_regions, ispec_dir + mySamOut_dir + "%7s_%4s_continuum_regions.txt" % (objName,chip))
    return star_continuum_regions


def SpecSave(star_spectrum):
    f = open(ispec_dir+mySamOut_dir+objName+"_CleanSpec.txt", "w")
    for s in star_spectrum:
        f.write('%9s\t%9s\t%9s\n' % (np.round(s['waveobs'],5),np.round(s['flux'],7), np.round(s['err'],7)))
    f.close()
def synthesize_spectrum(code="spectrum"):
    #--- Synthesizing spectrum -----------------------------------------------------
    # Parameters
    teff =  7554 #7554#
    logg =  1.04 #1.04#
    MH =  -0.92 #-0.92#
    alpha = ispec.determine_abundance_enchancements(MH)
    microturbulence_vel =  4.10 #4.10 #ispec.estimate_vmic(teff, logg, MH) # 1.07
    macroturbulence = ispec.estimate_vmac(teff, logg, MH) # 4.21
    vsini = 1.60 # Sun
    limb_darkening_coeff = 0.6
    resolution = 80000
    wave_step = 0.001

    # Wavelengths to synthesis
    #regions = ispec.read_segment_regions(ispec_dir + "/input/regions/fe_lines_segments.txt")
    regions = None
    wave_base = 340.0 # Magnesium triplet region
    wave_top = 700.0


    # Selected model amtosphere, linelist and solar abundances
    #model = ispec_dir + "/input/atmospheres/MARCS/"
    #model = ispec_dir + "/input/atmospheres/MARCS.GES/"
    #model = ispec_dir + "/input/atmospheres/MARCS.APOGEE/"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.APOGEE/"
    model = ispec_dir + "/input/atmospheres/ATLAS9.Castelli/"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kurucz/"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kirby/"

    atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.300_1100nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.1100_2400nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv6_atom_hfs_iso.420_920nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv6_atom_nohfs_noiso.420_920nm/atomic_lines.tsv"

    isotope_file = ispec_dir + "/input/isotopes/SPECTRUM.lst"

    # Load chemical information and linelist
    atomic_linelist_file = ispec_dir+mySamOut_dir+objName+"_LineList.txt"
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file)
    #atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=wave_base, wave_top=wave_top)
    #atomic_linelist = atomic_linelist[atomic_linelist['theoretical_depth'] >= 0.01] # Select lines that have some minimal contribution in the sun

    isotopes = ispec.read_isotope_data(isotope_file)

    if "ATLAS" in model:
        solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2009/stdatom.dat"
    else:
        # MARCS
        #solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.2007/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2005/stdatom.dat"
        solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2009/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Anders.1989/stdatom.dat"

    # Load model atmospheres
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)
    # Load SPECTRUM abundances
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)
    chemical_elements_file = ispec_dir + "/input/abundances/chemical_elements_symbols.dat"
    chemical_elements = ispec.read_chemical_elements(chemical_elements_file)
##--- Scaling all abundances -----------------------------------
    #for i,elem in enumerate(solar_abundances):
    #    if i>0 and i<82:
    #        elem['Abund'] += MH
    #solar_abundances['Abund'][55] += 3 #Ba
    #solar_abundances['Abund'][38] += 1 #Y
    #solar_abundances['Abund'][56] += 1.5 #La
    #solar_abundances['Abund'][57] += 1 #Ce
    #solar_abundances['Abund'][59] += 1 #Nd
    #solar_abundances['Abund'][58] += 1 #Pr
    #solar_abundances['Abund'][39] += 1 #Zr
    #solar_abundances['Abund'][5] += 1.5 #C
    #solar_abundances['Abund'][61] += 2 #Sm
    #solar_abundances['Abund'][62] += 2 #Eu
    #solar_abundances['Abund'][63] += 2 #Gd
    #solar_abundances['Abund'][65] += 2 #Dy
    #solar_abundances['Abund'][67] += 2 #Er
    #solar_abundances['Abund'][69] += 2 #Yb
    #solar_abundances['Abund'][70] += 2 #Lu
    #solar_abundances['Abund'][71] += 2 #Hf
    #solar_abundances['Abund'][73] += 2 #W
    #solar_abundances['Abund'][81] += 3.984 #Pb
    
    ## Custom fixed abundances
    #fixed_abundances = ispec.create_free_abundances_structure(["La", "Pr", "Ba","Y", "Ce", "Nd", "Zr", "Zn", "C", "Sm","Ti","V"], chemical_elements, solar_abundances)
    #fixed_abundances['Abund'] = [2-12.036,2-12.036,3-12.036,2-12.036,2-12.036,2-12.036,2-12.036,2-12.036,2-12.036,2-12.036, 2-12.036, 2-12.036] # Abundances in SPECTRUM scale (i.e., x - 12.0 - 0.036) and in the same order ["C", "N", "O"]
    fixed_abundances = ispec.create_free_abundances_structure(chemical_elements['symbol'][5:-33].tolist(), chemical_elements, solar_abundances) # from C to Pb
    #print(np.where(fixed_abundances['element'] == 'C')[0][0])
    #print(fixed_abundances['element'])
    fixed_abundances['Abund'] += MH # Scale to metallicity
    fixed_abundances['Abund'][76] += 3.984 #Pb abundance
    #fixed_abundances['Abund'][0] += 0.16 #C abundance
    ## No fixed abundances
    #fixed_abundances = None

    # Validate parameters
    if not ispec.valid_atmosphere_target(modeled_layers_pack, {'teff':teff, 'logg':logg, 'MH':MH, 'alpha':alpha}):
        msg = "The specified effective temperature, gravity (log g) and metallicity [M/H] \
                fall out of theatmospheric models."
        print(msg)

    # Prepare atmosphere model
    atmosphere_layers = ispec.interpolate_atmosphere_layers(modeled_layers_pack, {'teff':teff, 'logg':logg, 'MH':MH, 'alpha':alpha}, code=code)

    # Synthesis
    synth_spectrum = ispec.create_spectrum_structure(np.arange(wave_base, wave_top, wave_step))
    synth_spectrum['flux'] = ispec.generate_spectrum(synth_spectrum['waveobs'], \
            atmosphere_layers, teff, logg, MH, alpha, atomic_linelist, isotopes, solar_abundances, \
            fixed_abundances, microturbulence_vel = microturbulence_vel, \
            macroturbulence=macroturbulence, vsini=vsini, limb_darkening_coeff=limb_darkening_coeff, \
            R=resolution, regions=regions, verbose=1,
            code=code)        
    ##--- Save spectrum ------------------------------------------------------------
    logging.info("Saving spectrum...")
    synth_filename = "synthspecJ003643_Pb.txt" 
    ispec.write_spectrum(synth_spectrum, synth_filename)

if __name__ == '__main__':
    LinUpd()
    import ispec
    star_spectrum = ispec.read_spectrum(ispec_dir+mySamIn_dir+objName+".txt")
    star_spectrum, star_continuum_model = ContFitAndNorm(star_spectrum)
    star_spectrum = CosmicFilter(star_spectrum, star_continuum_model)
    SpecSave(star_spectrum)
    synthesize_spectrum(code="spectrum")
    pass
