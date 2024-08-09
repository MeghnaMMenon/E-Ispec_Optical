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
import ispec
import numpy as np
import logging
import multiprocessing
from multiprocessing import Pool

#--- iSpec directory -----------------------------------------------------------
objName = sys.argv[1]
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
#--- Definition of initial stellar parameters and other variables --------------
Aname, Ateff, Alogg, Amet, Avmic, Acomm = np.loadtxt(ispec_dir + mySamIn_dir + "StellarParam_%7s.txt" % objName, delimiter='\t', dtype=np.dtype([('name','U8'), ('teff',np.float64), ('logg',np.float64), ('met',np.float64), ('vmic',np.float64), ('comm','U5')]), skiprows=1, unpack=True)
index = (Aname==objName)
if np.any(index):
    initial_teff = float(Ateff[index]); initial_logg = float(Alogg[index]); initial_MH = float(Amet[index]); initial_vmic = float(Avmic[index])
else:
    initial_teff = 8250.0; initial_logg = 1.0; initial_MH = -1.18; initial_vmic = 2
initial_R = 57000.
star_spectrum = []; star_continuum_model = []; star_continuum_regions= []
estimated_snr = []; segments = []; star_linemasks = []



def ContFitAndNorm(star_spectrum):
    global initial_R
    logging.info("Fitting continuum...")
    model = "Splines" # "Polynomy"
    degree = 2
    from_resolution = initial_R
    
    # Strategy: Filter first median values and secondly MAXIMUMs in order to find the continuum
    order='median+max'
    logging.info("Median wavelength step: %.5f nm" % np.median(star_spectrum['waveobs'][1:]-star_spectrum['waveobs'][:-1]))
    median_wave_range=0.05 #3.*np.median(star_spectrum['waveobs'][1:]-star_spectrum['waveobs'][:-1])
    max_wave_range=0.25 #30.*np.median(star_spectrum['waveobs'][1:]-star_spectrum['waveobs'][:-1])
    nknots = None #int((np.max(star_spectrum['waveobs']) - np.min(star_spectrum['waveobs'])) / max_wave_range)+1 #len(segments) # Automatic: 1 spline every 5 nm
    
    star_continuum_model = ispec.fit_continuum(star_spectrum, from_resolution=initial_R, \
                                nknots=nknots, degree=degree, \
                                median_wave_range=median_wave_range, \
                                max_wave_range=max_wave_range, \
                                model=model, order=order, \
                                automatic_strong_line_detection=True, \
                                strong_line_probability=0.5, \
                                use_errors_for_fitting=True)
    #--- Normalise -------------------------------------------------------------
    star_spectrum = ispec.normalize_spectrum(star_spectrum, star_continuum_model, consider_continuum_errors=False)
    return(star_spectrum)
    
def RVCorr(star_spectrum):
    #--- Radial Velocity determination with template ---------------------------
    logging.info("Radial velocity determination with mask...")
    # - Read synthetic template
    smooth_spectrum = ispec.convolve_spectrum(star_spectrum, 50000., from_resolution=initial_R)
    #f = open(ispec_dir+mySamOut_dir+objName+"_degraded.txt", "w")
    #f.write('waveobs\tflux\terr\n')
    #for s in star_spectrum:
    #    f.write('%9s\t%9s\t%9s\n' % (np.round(s['waveobs'],5),np.round(s['flux'],7), np.round(s['err'],7)))
    #f.close()
    wfilter = ispec.create_wavelength_filter(smooth_spectrum, wave_base=450.0, wave_top=650.0)
    smooth_spectrum = smooth_spectrum[wfilter]
    template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Arcturus.372_926nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Sun.372_926nm/template.txt.gz")
    logging.info("Radial velocity determination with Arcturus template...")
    models, ccf = ispec.cross_correlate_with_template(smooth_spectrum, template,\
            lower_velocity_limit=-120., upper_velocity_limit=120.,\
            velocity_step=0.5,fourier=True)
    # Plotting CCF (to exclude those contaminated by the shocks)
    vel = [c[0] for c in ccf] # Velocity
    val = [c[1] for c in ccf] # CCF values
    from matplotlib import pyplot as plt
    plt.rcParams["font.family"] = "Times New Roman"
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Normalised CCF')
    plt.title(objName)
    plt.plot(vel, val)
    plt.grid()
    plt.savefig(ispec_dir+mySamOut_dir+objName+"_RV.pdf")
    #plt.show()
    # Number of models represent the number of components
    components = len(models)
    # First component:
    rv = np.round(models[0].mu(), 2) # km/s
    rv_err = np.round(models[0].emu(), 2) # km/s
    
    #--- Radial Velocity correction --------------------------------------------
    #logging.info("Radial velocity correction... %.2f +/- %.2f" % (rv, rv_err))
    #star_spectrum = ispec.correct_velocity(star_spectrum, rv)
    return(star_spectrum, rv, rv_err)
    
def CosmicFilter(spec, star_continuum_model):
    # Spectrum should already be normalised
    step = spec['waveobs'][len(spec)//2+1] - spec['waveobs'][len(spec)//2]
    cosmics = ispec.create_filter_cosmic_rays(spec,star_continuum_model,\
              resampling_wave_step=step, window_size=15, \
              variation_limit=0.5)
    spec = spec[~cosmics]
    return(spec)

def SNRErrCalc(star_spectrum):
    #--- Estimate SNR from flux and errors from SNR ----------------------------
    logging.info("Estimating SNR from fluxes (negative flux values will be set to zero)...")
    zeroF = np.where(star_spectrum['flux']<0.0)
    star_spectrum['flux'][zeroF] = 0.0
    num_points = 3
    blue_spec = star_spectrum[(star_spectrum['waveobs'] >= 662.) & (star_spectrum['waveobs'] <= 662.4)]
    green_spec = star_spectrum[(star_spectrum['waveobs'] >= 663.9) & (star_spectrum['waveobs'] <= 664.2)]
    red_spec = star_spectrum[(star_spectrum['waveobs'] >= 675.8) & (star_spectrum['waveobs'] <= 676.4)]
    blue_dump = blue_spec[blue_spec['flux']>0.]
    print('SNR: ', np.nanmedian(blue_dump['flux'])/np.nanstd(blue_dump['flux']), np.nanmedian(green_spec['flux'])/np.nanstd(green_spec['flux']), np.nanmedian(red_spec['flux'])/np.nanstd(red_spec['flux']))
    blue_snr = ispec.estimate_snr(blue_spec['flux'], num_points=num_points)
    green_snr = ispec.estimate_snr(green_spec['flux'], num_points=num_points)
    red_snr = ispec.estimate_snr(red_spec['flux'], num_points=num_points)
    blue_spec = star_spectrum[star_spectrum['waveobs'] <= 465.]
    green_spec = star_spectrum[(star_spectrum['waveobs'] >= 465.) & (star_spectrum['waveobs'] <= 580.)]
    red_spec = star_spectrum[star_spectrum['waveobs'] >= 580.]
    star_spectrum['err'] = np.concatenate((blue_spec['flux']/30., green_spec['flux']/90., red_spec['flux']/90.))
    snr = np.array([blue_snr,green_snr,red_snr])
    return(snr)
    
def ListCreation(wave_base, wave_top):
    #--- Calculate theoretical equivalent widths and depths for a linelist -----
    global initial_teff, initial_logg, initial_MH, initial_vmic
    logging.info("CREATING A LINE LIST FOR THE FOLLOWING AP: %4.0f, %1.1f, %1.2f, %1.1f" % (initial_teff, initial_logg, initial_MH, initial_vmic))
    alpha = ispec.determine_abundance_enchancements(initial_MH)
    model = ispec_dir + "/input/atmospheres/MARCS.GES/"
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)
    atmosphere_layers = ispec.interpolate_atmosphere_layers(modeled_layers_pack, {'teff':initial_teff, 'logg':initial_logg, 'MH':initial_MH, 'alpha':alpha})
    isotopes = ispec.read_isotope_data(ispec_dir + "/input/isotopes/SPECTRUM.lst")
    atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.300_1100nm/atomic_lines.tsv"
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=wave_base, wave_top=wave_top)
    chemical_elements = ispec.read_chemical_elements(ispec_dir + "/input/abundances/chemical_elements_symbols.dat")
    solar_abundances = ispec.read_solar_abundances(ispec_dir + "/input/abundances/Asplund.2009/stdatom.dat")
    #solar_abundances['Abund'][solar_abundances['code']==chemical_elements['atomic_num'][chemical_elements['symbol']=='C']] += 1.
    #solar_abundances['Abund'][solar_abundances['code']==chemical_elements['atomic_num'][chemical_elements['symbol']=='N']] += 1.
    #solar_abundances['Abund'][solar_abundances['code']==chemical_elements['atomic_num'][chemical_elements['symbol']=='O']] += 1.
    #CNO_mask = atomic_linelist['element']=='C 1'
    #CNO_mask = np.logical_or(CNO_mask, atomic_linelist['element']=='N 1')
    #CNO_mask = np.logical_or(CNO_mask, atomic_linelist['element']=='O 1')
    #atomic_linelist = atomic_linelist[CNO_mask]
    new_atomic_linelist = ispec.calculate_theoretical_ew_and_depth(atmosphere_layers, \
            initial_teff, initial_logg, initial_MH+1., alpha, \
            atomic_linelist, isotopes, solar_abundances, microturbulence_vel=initial_vmic, \
            verbose=1, gui_queue=None, timeout=900)
    #new_atomic_linelist = new_atomic_linelist[new_atomic_linelist['theoretical_ew']>5.]
    ispec.write_atomic_linelist(new_atomic_linelist, linelist_filename=ispec_dir + mySamOut_dir + objName + "_LineList.txt")
    
def LineFit(star_spectrum, star_continuum_model, rv, mode='tweak'):
    #--- Reading required files ------------------------------------------------
    atomic_linelist_file = ispec_dir+mySamOut_dir+objName+"_LineList.txt"
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=np.min(star_spectrum['waveobs']), wave_top=np.max(star_spectrum['waveobs']))
    #atomic_linelist = atomic_linelist[atomic_linelist['theoretical_depth']>0.]
    
    #telluric_linelist_file = ispec_dir + "/input/linelists/CCF/Synth.Tellurics.500_1100nm/mask.lst"
    #telluric_linelist = ispec.read_telluric_linelist(telluric_linelist_file, minimum_depth=0.01)
    #models, ccf = ispec.cross_correlate_with_mask(star_spectrum, telluric_linelist, \
    #        lower_velocity_limit=-300., upper_velocity_limit=300., \
    #        velocity_step=0.5, mask_depth=0.01, \
    #        fourier = False, only_one_peak = True)
    #if not models:
    #    logging.info("TELLURIC VELOCITY NOT FOUND. TAKING RELATIVE VELOCITY WITH NEGATIVE SIGN INSTEAD")
    #    vel_telluric = -rv
    #    vel_telluric_err = rv_err
    #else:
    #    vel_telluric = np.round(models[0].mu(), 2) # km/s
    #    vel_telluric_err = np.round(models[0].emu(), 2) # km/s
    vel_telluric = -rv; vel_telluric_err = rv_err
    
    # Line regions
    if mode=='seek':
        smooth_R = initial_R/1.25; ID = 'element'
        smooth_spectrum = ispec.convolve_spectrum(star_spectrum, smooth_R, from_resolution=initial_R)
        #####--- Find linemasks ------------------------------------------------
        line_regions = ispec.find_linemasks(star_spectrum, star_continuum_model,\
                                atomic_linelist=atomic_linelist, \
                                max_atomic_wave_diff = 0.005, \
                                telluric_linelist=None, \
                                vel_telluric=vel_telluric, \
                                minimum_depth=0.1, maximum_depth=0.5, \
                                smoothed_spectrum=smooth_spectrum, \
                                check_derivatives=False, \
                                discard_gaussian=False, discard_voigt=True, \
                                closest_match=False)
        # Discard bad masks
        flux_peak = smooth_spectrum['flux'][line_regions['peak']]
        flux_base = smooth_spectrum['flux'][line_regions['base']]
        flux_top = smooth_spectrum['flux'][line_regions['top']]
        bad_mask = np.logical_or(line_regions['wave_peak'] <= line_regions['wave_base'], line_regions['wave_peak'] >= line_regions['wave_top'])
        bad_mask = np.logical_or(bad_mask, flux_peak >= flux_base)
        bad_mask = np.logical_or(bad_mask, flux_peak >= flux_top)
        bad_mask = np.logical_or(bad_mask, line_regions['element']=='')
        bad_mask = np.logical_or(bad_mask, line_regions['molecule']!='F')
        line_regions = line_regions[~bad_mask]
        # Leave only Fe 1 and Fe 2 lines (and others if needed)
        FeCNO_mask = np.logical_or(line_regions['element']=='Fe 1', line_regions['element']=='Fe 2')
        FeCNO_mask = np.logical_or(FeCNO_mask, line_regions['element']=='C 1')
        FeCNO_mask = np.logical_or(FeCNO_mask, line_regions['element']=='N 1')
        FeCNO_mask = np.logical_or(FeCNO_mask, line_regions['element']=='O 1')
        FeCNO_mask = np.logical_or(FeCNO_mask, line_regions['element']=='S 1')
        FeCNO_mask = np.logical_or(FeCNO_mask, line_regions['element']=='Zn 1')
        line_regions = line_regions[FeCNO_mask] # or line_regions[~FeCNO_mask]
        line_regions = line_regions[line_regions['theoretical_ew'] > 0.]
    
    if mode=='tweak' or mode=='pick':
        line_regions = ispec.read_line_regions(ispec_dir + mySamIn_dir + "FinalLinelist_%7s.txt" % objName); ID = 'note'

    ids = [l[ID] for l in line_regions]
    ids_counted = {item:ids.count(item) for item in ids}
    logging.info("LINES READ: " + str(ids_counted))
    logging.info("TOTAL NUMBER: " + str(len(line_regions)))
    #--- Fit lines -------------------------------------------------------------
    logging.info("Fitting lines...")
    #line_regions = ispec.adjust_linemasks(star_spectrum, line_regions, max_margin=0.5)
    linemasks = ispec.fit_lines(line_regions, star_spectrum, star_continuum_model, \
                atomic_linelist = atomic_linelist, \
                max_atomic_wave_diff = 0.005, \
                telluric_linelist = None, \
                smoothed_spectrum = None, \
                check_derivatives = False, \
                vel_telluric = None, discard_gaussian=False, \
                discard_voigt=True, \
                free_mu=True, crossmatch_with_mu=False, closest_match=False)
    line_regions.sort(order='wave_peak'); linemasks.sort(order='wave_peak')
    linemasks = LineFitFilt(line_regions, linemasks, ID)
    LineFitPlot(star_spectrum, linemasks, mode)
    #LineComparPlot(linemasks)
    return(linemasks)

def LineFitFilt(line_regions, linemasks, ID):
    #############################################################################    
    f = open(ispec_dir+mySamOut_dir+objName+"_lineregs.txt", "w")
    for l in linemasks:
        f.write('%.4f\t%.4f\t%.4f\t%s\n' % (l['wave_peak'], l['wave_base'], l['wave_top'], l['element']))
    f.close()
    #############################################################################      
    ispec.write_line_regions(linemasks, ispec_dir+mySamOut_dir+objName+"_LineFit.txt", extended=True)
    f = open(ispec_dir+mySamOut_dir+objName+"_LineFit_short.txt", "w")
    for l in line_regions:
        f.write('%s\t%.4f\n' % (l['note'], l['wave_peak']))
    f.close()
    
    # Discard lines that are not cross matched with the same original element stored in the note. Uncomment the below line if the linelist is not manually cleaned up.
    linemasks = linemasks[linemasks['element'] == line_regions[ID]]
    print("After element=note")
    print(len(linemasks)) #8686
    #############################################################################    
    f = open(ispec_dir+mySamOut_dir+objName+"_linemask_AfterElementNoteComparison", "w")
    for l in linemasks:
        f.write('%.4f\t%.4f\t%.4f\t%s\n' % (l['wave_peak'], l['wave_base'], l['wave_top'], l['element']))
    f.close()
    #############################################################################      
    #Nothing lost until this step. Check from here tomorow.
    
    # Select lines that have some minimal contribution in pAGBs
#    atomic_linelist = atomic_linelist[atomic_linelist['theoretical_depth'] >= 0.01]
    # Exclude lines that have not been successfully cross matched with the atomic data
    # because we cannot calculate the chemical abundance (it will crash the corresponding routines)

    linemasks = linemasks[linemasks['wave_nm']!=0.]
    
    #############################################################################    
    f = open(ispec_dir+mySamOut_dir+objName+"_linemask_AfterWaveNotZero", "w")
    for l in linemasks:
        f.write('%.4f\t%.4f\t%.4f\t%s\n' % (l['wave_peak'], l['wave_base'], l['wave_top'], l['element']))
    f.close()
    #############################################################################  
    linemasks = linemasks[linemasks['ew']>5.]#5
    linemasks = linemasks[linemasks['ew']<350.]#250
    linemasks.sort(order='spectrum_moog_species')
    #############################################################################    
    f = open(ispec_dir+mySamOut_dir+objName+"_linemask_AfterEWFilter", "w")
    for l in linemasks:
        f.write('%.4f\t%.4f\t%.4f\t%s\n' % (l['wave_peak'], l['wave_base'], l['wave_top'], l['element']))
    f.close()
    #############################################################################     
    ispec.write_line_regions(linemasks, ispec_dir+mySamOut_dir+objName+ "_LineFit.txt", extended=True)
    f = open(ispec_dir+mySamOut_dir+objName+"_LineFit_short.txt", "w")
    for l in linemasks:
        f.write('%s\t%.4f\t%.2f\t%.2f\n' % (l['element'], l['wave_nm'], l['ew'], l['ew_err']))
    f.close()
    f = open(ispec_dir+mySamOut_dir+objName+"_FitGoodness.txt", "w")
    f.write('lambda (nm)\tArea Ratio (%)\n')
    dev = 0.
    goodLnNum = 0
#    for l in linemasks:
#        spec = star_spectrum[l['base']:l['top']]
#        obs = spec['flux']
#        fit = 1.+l['A']*np.exp(-(spec['waveobs']-l['mu'])**2/(2.*l['sig']**2))
#        print('%3.4f, %1.3f' % (l['wave_peak'], np.sum(np.abs(obs-fit))/l['depth']))
#        ratio = np.sum(np.abs(obs-fit))*(spec['waveobs'][1]-spec['waveobs'][0])/l['ew']*1.e+4
#        if np.abs(ratio) > 0.5:
#            l['ew'] = 0.
#        else:
#            goodLnNum += 1
#            dev += np.abs(ratio)
#        f.write('%4s\t%3.4f\t%.3f\n' % (l['element'], l['wave_peak'], ratio*100.))
#    f.write('OVERALL:\t%.2f (%3i)' % (np.round(dev*100.,2), goodLnNum))
#    logging.info("THE OVERALL AREA DEVIATION IS " + str(np.round(dev*100.,2)) + " BASED ON " + str(goodLnNum) + " LINES")
    f.close()
    
    #############################################################################    
    f = open(ispec_dir+mySamOut_dir+objName+"_linemask_AfterAreaDetermination", "w")
    for l in linemasks:
        f.write('%.4f\t%.4f\t%.4f\t%s\n' % (l['wave_peak'], l['wave_base'], l['wave_top'], l['element']))
    f.close()
    #############################################################################  
    print(len(linemasks))
    linemasks = linemasks[linemasks['ew']>0.]
    print(len(linemasks))
    #############################################################################    
    f = open(ispec_dir+mySamOut_dir+objName+"_linemask_AfterAllFilters", "w")
    f.write('wave_peak\twave_base\twave_top\tnote\n')
    for l in linemasks:
        f.write('%.4f\t%.4f\t%.4f\t%s\n' % (l['wave_peak'], l['wave_base'], l['wave_top'], l['element']))
    f.close()
    #############################################################################  
    ids = [l['element'] for l in linemasks]
    ids_counted = {item:ids.count(item) for item in ids}
    logging.info("LINES IDENTIFIED: " + str(ids_counted))
    logging.info("TOTAL NUMBER: " + str(len(linemasks)))
    return(linemasks)

def LineFitPlot(star_spectrum, linemasks, mode):
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    if mode=='tweak':
        addon = "WithSynth"
    else:
        addon = ""
    pdf = PdfPages(ispec_dir+mySamOut_dir+objName+"_FittedLines"+addon+".pdf")
    atomic_linelist_file = ispec_dir+mySamOut_dir+objName+"_LineList.txt"
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=np.min(star_spectrum['waveobs']), wave_top=np.max(star_spectrum['waveobs']))
    for i in range(len(linemasks)):
        logging.info('PLOTTING LINE ' + str(i+1) + '/' + str(len(linemasks)))
        numGap = 100
        spec = star_spectrum[linemasks['peak'][i]-numGap:linemasks['peak'][i]+numGap]
        gauss = 1.+linemasks['A'][i]*np.exp(-(spec['waveobs']-linemasks['mu'][i])**2/(2.*linemasks['sig'][i]**2))
        if len(spec['waveobs'])<numGap:
            continue
        step = spec['waveobs'][numGap]-spec['waveobs'][numGap-1]
        from_x = numGap - int(6.*linemasks['sig'][i]/step)
        to_x = numGap + int(6.*linemasks['sig'][i]/step)
        wave_filter_custom = (spec['waveobs'] >= linemasks['wave_base'][i]) & (spec['waveobs'] <= linemasks['wave_top'][i])
        specline = spec[wave_filter_custom]
        lineEndsX = [specline['waveobs'][0], specline['waveobs'][-1]]
##        lineEndsX = [linemasks['wave_base'][i]-linemasks['wave_peak'][i]+100, linemasks['wave_top'][i]-linemasks['wave_peak'][i]+100]
#        lineEndsY = [specline['flux'][0], specline['flux'][-1]]
        lineEndsY = [1., 1.]
        #cont_a = (specline['flux'][-1]-specline['flux'][0])/(specline['waveobs'][-1]-specline['waveobs'][0])
        #cont_b = (specline['waveobs'][-1]*specline['flux'][0]-specline['waveobs'][0]*specline['flux'][-1])/(specline['waveobs'][-1]-specline['waveobs'][0])
        cont_a = 0.; cont_b = 1.
        continuum_custom = cont_a*spec['waveobs'] + cont_b
        
        solar_abundances = ispec.read_solar_abundances(ispec_dir + "/input/abundances/Asplund.2009/stdatom.dat")
        chemical_elements = ispec.read_chemical_elements(ispec_dir + "/input/abundances/chemical_elements_symbols.dat")
        ID = chemical_elements['atomic_num'][chemical_elements['symbol']==linemasks['element'][i][:-2]]
        global initial_MH
        abund = solar_abundances['Abund'][solar_abundances['code']==ID]+12.036+initial_MH
        
        fig = plt.figure(figsize=(11,8))
        ax = fig.add_subplot(1,1,1)
        plt.xlim([spec['waveobs'][0]+0.01,spec['waveobs'][-1]-0.01])
        plt.ylim([0.2,1.2])
        plt.xlabel('$\lambda$ (nm)')
        plt.title('%s (%i/%i), EW = %.1f mA, initial guess EW = %.1f mA' % (linemasks['element'][i],i+1,len(linemasks),linemasks['ew'][i], linemasks['theoretical_ew'][i]))
        ax.plot(spec['waveobs'], spec['flux'], '-k', label='Observed spectrum')
        
        # Add pointers with elements to lines with EW>5 mA
        significant_linelist = atomic_linelist[atomic_linelist['wave_nm']>spec['waveobs'][0]]
        significant_linelist = significant_linelist[significant_linelist['wave_nm']<spec['waveobs'][-1]]
        significant_linelist = significant_linelist[significant_linelist['theoretical_ew']/linemasks['ew'][i]>0.05]
        for line in significant_linelist:
            plt.annotate(line['element'], xy=(line['wave_nm'], 1.02), xytext=(line['wave_nm'], 1.12), rotation=90, ha='center', fontsize=15, arrowprops=dict(arrowstyle="-", facecolor='black', lw=2))
        
        if mode=='tweak':
            synth0 = SynthSpec(spec, linemasks[i], -10.)
            synth = SynthSpec(spec, linemasks[i], abund)
            ax.plot(synth['waveobs'], synth['flux'], '--g', label='A(%s)=%.2f' % (linemasks['element'][i][:-2], abund))
            ax.plot(synth0['waveobs'], synth0['flux'], '-.m', label='A(%s)=-10.00' % linemasks['element'][i][:-2])
        
        #ax.plot(lineEndsX, lineEndsY, 'co', ms=5)
        #ax.plot(lineEndsX, lineEndsY, 'c-', lw=3)
        ax.plot(spec['waveobs'][from_x:to_x], gauss[from_x:to_x], '-r', label='Gaussian fit')
#        ax.axvline(linemasks['wave_base'][i], c='yellow', zorder=0)
#        ax.axvline(linemasks['wave_top'][i], c='yellow', zorder=0)
        ax.axvline(linemasks['wave_nm'][i], ymin=0., ymax=0.8, c='red', ls=':', zorder=1)
        #ax.axvline(lineEndsX[0], c='olive', ls=':', zorder=1)
        #ax.axvline(lineEndsX[-1], c='olive', ls=':', zorder=1)
        ax.axhline(1., c='gray', ls=':', zorder=0, label='Continuum')
        ax.fill_between(lineEndsX, y1=0., y2=1., color='none', edgecolor='olive', hatch='\\\\\\', zorder=0., label='Line region') #np.min(specline['flux'])
        ax.fill_between(spec['waveobs'][from_x:to_x], gauss[from_x:to_x], continuum_custom[from_x:to_x], color='none', edgecolor='cyan', hatch='///', zorder=0, label='Line area (EW)')
        ax.legend(ncol=2, loc='lower left')
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
        plt.close()
    pdf.close()

def LineComparPlot(linemasks):
    import matplotlib.pyplot as plt
    color_dict = {key: np.where(linemasks['element']==key)[0][0] for key in linemasks['element']}
    elems = []; index=[]
    Slamb, Sew, Sel = np.loadtxt(ispec_dir + mySamIn_dir + objName + "_ew_lines.txt", delimiter=',', dtype=np.dtype([('waveobs','U8'), ('ew','U5'), ('elem','U5')]), unpack=True)
    Slamb = np.asarray(Slamb).astype(float); Sew = np.asarray(Sew).astype(float); Sel = np.asarray(Sel)
    index = [np.min(abs(l-linemasks['wave_nm']))<0.005 for l in Slamb]
    Slamb = Slamb[index]; Sew = Sew[index]; Sel = Sel[index]
    colormap = plt.cm.gist_ncar
    colorst = [colormap(i) for i in np.linspace(0, 0.9,len(color_dict))]
    fileRatios = open(ispec_dir+mySamOut_dir+objName+"_ComparEWs.txt", "w")
    fileRatios.write('element\twavelength\tEWSp\tEWlit\tdEW/EWlit (%)\n')
    dev = 0.
    for i in range(len(color_dict)):
        dump = linemasks[linemasks['element'] == list(color_dict)[i]]
        litEW = Sew[Sel == list(color_dict)[i]]
        plt.scatter(litEW, (dump['ew']-litEW)/litEW*100., s=5, color=colorst[i], label=list(color_dict)[i] + ': ' + str(len(linemasks[linemasks['element'] == list(color_dict)[i]])), marker=i%12, zorder=1)
#        plt.errorbar(litEW, dump['ew'], yerr=dump['ew_err'], fmt='none', lw=0, ecolor='lightgray', elinewidth=0.5, zorder=0)
        for k in range(len(dump)):
            fileRatios.write('%4s\t%3.4f\t%3.1f\t%3.1f\t%2.2f\n' % (dump['element'][k],dump['wave_nm'][k],dump['ew'][k],litEW[k],(dump['ew'][k]-litEW[k])/litEW[k]*100.))
            dev += ((dump['ew'][k]-litEW[k])/litEW[k])**2
    fileRatios.write('Overall deviation = %3.4f per cent' % (dev*100.))
    fileRatios.close()
    logging.info('OVERALL DEVIATION IS' + str(dev*100.) + '%')
    x = np.linspace(np.min(Sew), np.max(Sew), 10)

#    x = linemasks['depth']
#    y1 = (linemasks['top']-linemasks['peak'])**2
#    y2 = (linemasks['peak']-linemasks['base'])**2
#    model1 = np.polyfit(x, y1, 1)
#    model1_fn = np.poly1d(model1)
#    model2 = np.polyfit(x, y2, 1)
#    model2_fn = np.poly1d(model2)
#    plt.scatter(x, y1, s=5, color='magenta', zorder=1)
#    plt.scatter(x, y2, s=5, color='cyan', zorder=1)
#    plt.plot(x, model1_fn(x), 'magenta', label='%3.2f*x+%3.2f' % (model1[0], model1[1]))
#    plt.plot(x, model2_fn(x), 'cyan', label='%3.2f*x+%3.2f' % (model2[0], model2[1]))
    
#    plt.plot(x,1,'k')
#    plt.plot(x,1.33*np.sqrt(2),'gray', label='$\pm$33%')
#    plt.plot(x,1.33/np.sqrt(2),'gray')
    plt.axhline(0., c='black', ls='--', zorder=0)
    plt.axhline(50., c='gray', ls='--', zorder=0, label='$\pm$50%')
    plt.axhline(-50., c='gray', ls='--', zorder=0)
    plt.xlabel("$EW_{\\rm lit}$ (mA)")
    plt.ylabel("$(EW_{\\rm exp}-EW_{\\rm lit})/EW_{\\rm lit}$ (%)")
    plt.ylim([-200.,200.])
    plt.title(objName)
    plt.xscale('log')
#    plt.loglog()
    plt.legend(ncol=2, prop={'size': 5})
    plt.savefig(ispec_dir+mySamOut_dir+objName+"_ComparEWs.pdf", bbox_inches='tight', pad_inches=0)
#    plt.show()
    plt.close()
    
def SynthSpec(star_spectrum, regions, abund=0., code='turbospectrum'):
    global initial_teff, initial_logg, initial_MH, initial_vmic, initial_R
    initial_alpha = ispec.determine_abundance_enchancements(initial_MH)
    macroturbulence = ispec.estimate_vmac(initial_teff, initial_logg, initial_MH)
    max_iterations = 10
    model = ispec_dir + "input/atmospheres/ATLAS9.Castelli/" #MARCS.GES / ATLAS9.KuruczODFNEW
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)
    atmosphere_layers = ispec.interpolate_atmosphere_layers(modeled_layers_pack, {'teff':initial_teff, 'logg':initial_logg, 'MH':initial_MH, 'alpha':initial_alpha})
    isotopes = ispec.read_isotope_data(ispec_dir + "/input/isotopes/SPECTRUM.lst")
    solar_abundances = ispec.read_solar_abundances(ispec_dir + "/input/abundances/Asplund.2009/stdatom.dat")
    atomic_linelist_file = ispec_dir+mySamOut_dir+objName+"_LineList.txt"
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=np.min(star_spectrum['waveobs']), wave_top=np.max(star_spectrum['waveobs']))
    chemical_elements_file = ispec_dir + "/input/abundances/chemical_elements_symbols.dat"
    chemical_elements = ispec.read_chemical_elements(chemical_elements_file)
    if abund!=0.:
        fixed_abundances = ispec.create_free_abundances_structure([regions['element'][:-2]], chemical_elements, solar_abundances)
        fixed_abundances['Abund'] = [abund-12.036]
    else:
        fixed_abundances=None
    # Synthesis
    synth_spectrum = ispec.create_spectrum_structure(star_spectrum['waveobs'])
    synth_spectrum['flux'] = ispec.generate_spectrum(synth_spectrum['waveobs'], \
            atmosphere_layers, initial_teff, initial_logg, initial_MH, initial_alpha, 
            atomic_linelist, isotopes, solar_abundances, \
            fixed_abundances=fixed_abundances, microturbulence_vel = initial_vmic, \
            macroturbulence=macroturbulence, vsini=0., limb_darkening_coeff=0.6, \
            R=initial_R, regions=None, verbose=1, code=code)
    return(synth_spectrum)
    
def EWparam(star_spectrum, star_continuum_model, linemasks, code='moog', mode='default'):
    #--- Model spectra from EW -------------------------------------------------
    global initial_teff, initial_logg, initial_MH, initial_vmic
    initial_alpha = ispec.determine_abundance_enchancements(initial_MH)
    max_iterations = 10

    #--- Model spectra ---------------------------------------------------------
    logging.info("Last initializations...")
    # Selected model amtosphere, linelist and solar abundances
    model = ispec_dir + "input/atmospheres/ATLAS9.Castelli/" #MARCS.GES / ATLAS9.KuruczODFNEW

    # Load SPECTRUM abundances
    solar_abundances_file = ispec_dir + "input/abundances/Asplund.2009/stdatom.dat"
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)
    
    # Load model atmospheres
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)

    # Validate parameters
    if not ispec.valid_atmosphere_target(modeled_layers_pack, {'teff':initial_teff, 'logg':initial_logg, 'MH':initial_MH, 'alpha':initial_alpha}):
        msg = "The specified effective temperature, gravity (log g) and metallicity [M/H] \
                fall out of the atmospheric models."
        print(msg)
    #print(linemasks)
    # Reduced equivalent width
    # Filter too weak/strong lines
    # * Criteria presented in paper of GALA
    #efilter = np.logical_and(linemasks['ewr'] >= -5.8, linemasks['ewr'] <= -4.65)
    efilter = np.logical_and(linemasks['ewr'] >= -6.0, linemasks['ewr'] <= -4.3)
    # Filter high excitation potential lines
    # * Criteria from Eric J. Bubar "Equivalent Width Abundance Analysis In Moog"
    efilter = np.logical_and(efilter, linemasks['lower_state_eV'] <= 5.0)
    efilter = np.logical_and(efilter, linemasks['lower_state_eV'] >= 0.5)
    ## Filter also bad fits
    efilter = np.logical_and(efilter, linemasks['rms'] < 1.00)
    # no flux
    noflux = star_spectrum['flux'][linemasks['peak']] < 1.0e-10
    efilter = np.logical_and(efilter, np.logical_not(noflux))
    unfitted = linemasks['fwhm'] == 0
    efilter = np.logical_and(efilter, np.logical_not(unfitted))
    linemasks = linemasks[efilter]
    print(len(linemasks))
#############################################################################    
    f = open(ispec_dir+mySamOut_dir+objName+"_Litlast.txt", "w")
    for l in linemasks:
        f.write('%.4f\t%.4f\t%.4f\t%.4f\t%s\n' % (l['wave_nm'], l['wave_peak'], l['wave_base'], l['wave_top'], l['element']))
    f.close()
#############################################################################
    #FeIIlineMask = linemasks['wave_nm']==624.7557
    #linemasks['ew'][FeIIlineMask] -= 32.
    results = ispec.model_spectrum_from_ew(linemasks, modeled_layers_pack, \
                        solar_abundances, initial_teff, initial_logg, initial_MH, initial_alpha, initial_vmic, \
                        free_params=["teff", "logg", "vmic"], \
                        adjust_model_metalicity=True, \
                        max_iterations=max_iterations, \
                        enhance_abundances=True, \
                        #outliers_detection = "robust", \
                        #outliers_weight_limit = 0.90, \
                        outliers_detection = "sigma_clipping", \
                        sigma_level = 10., \
                        tmp_dir = None, \
                        code=code) #"teff", "logg", "vmic"
    params, errors, status, x_over_h, selected_x_over_h, fitted_lines_params, used_linemasks = results
    #print(len(linemasks), len(x_over_h), len(used_linemasks))
    initial_teff = params['teff']; initial_logg = params['logg']; initial_MH = params['MH']; initial_vmic = params['vmic']
    ##--- Plotting the constraints' plots --------------------------------------
    CnstrPlot(linemasks, x_over_h)
    
    ##--- Save results ---------------------------------------------------------
    logging.info("Saving results...")
    f = open(ispec_dir+mySamOut_dir+objName+"_res_StePar.txt", "w")
    f.write('Teff\teTeff\tlogg\telogg\tMH\teMH\tvmic\tevmic\n')
    f.write('%4.0f\t%4.0f\t%1.2f\t%1.2f\t%1.2f\t%1.2f\t%1.2f\t%1.2f\n' % (params['teff'], errors['teff'], params['logg'], errors['logg'], params['MH'], errors['MH'], params['vmic'], errors['vmic']))
    f.close()

def CnstrPlot(linemasks, x_over_h):
    import matplotlib.pyplot as plt
    Fe_line_regions = [line for line in linemasks if (line[0]=='Fe 1' or line[0]=='Fe 2')]
    Fe_line_regions = np.array(Fe_line_regions)
    #print(len(Fe_line_regions))
    #print(Fe_line_regions)
    idx = np.isfinite(x_over_h)
    Fe_line_regions = Fe_line_regions[idx]; x_over_h = x_over_h[idx]
    ind1 = np.where(Fe_line_regions['element']=='Fe 1')[0]
    ind2 = np.where(Fe_line_regions['element']=='Fe 2')[0]
    #print(ind1, ind2)
    
    f = open(ispec_dir+mySamOut_dir+objName+"_FinalValuesforPlotting", "w")
    for l in Fe_line_regions:
        f.write('%.4f\t%.4f\t%.2f\t%.2f\t%s\n' % (l['wave_peak'], l['lower_state_eV'], l['ewr'], l['theoretical_ew'], l['element']))
    f.close()
    for i in range(len(x_over_h)):
        print('%s: CWL=%.5f, [Fe/H]=%.2f, EW=%.2f, thEW=%.2f, rEW=%.2f, eV=%.2f' % (Fe_line_regions['element'][i], Fe_line_regions['wave_peak'][i], x_over_h[i], Fe_line_regions['ew'][i], Fe_line_regions['theoretical_ew'][i], Fe_line_regions['ewr'][i], Fe_line_regions['lower_state_eV'][i]))
    
    ## v_mic correction
    plt.xlim([-6.,-4.2])
    plt.ylim([np.nanmean(x_over_h)-1,np.nanmean(x_over_h)+1])
    plt.xlabel("reduced EW")
    plt.ylabel("[Fe/H]")
    plt.title('%s, $v_{mic}$ correction' % objName)
    plt.grid(True, zorder=0)
    coef = np.polyfit(Fe_line_regions['ewr'],x_over_h,1)
    poly1d_fn = np.poly1d(coef)
    x = np.linspace(-6., -4.2, 10)
    plt.plot(x, poly1d_fn(x), '-k', zorder=1)
#    plt.hlines(np.nanmean(x_over_h), -6, -4.2, zorder=0)
    plt.scatter(Fe_line_regions['ewr'][ind1], x_over_h[ind1], s=10, c='olive', zorder=2, label='Fe 1: %3i lines' % len(x_over_h[ind1]))
    plt.scatter(Fe_line_regions['ewr'][ind2], x_over_h[ind2], s=10, c='cyan', zorder=2, label='Fe 2: %3i lines' % len(x_over_h[ind2]))
    plt.legend()
    plt.savefig(ispec_dir+mySamOut_dir+objName+"_res_SlopeEqu.pdf", bbox_inches='tight', pad_inches=0)
    plt.close()
    
    ## Excitation equilibrium
    plt.xlim([.5,5.])
    plt.ylim([np.nanmean(x_over_h)-1,np.nanmean(x_over_h)+1])
    plt.xlabel("lower state (eV)")
    plt.ylabel("[Fe/H]")
    plt.title('%s, excitation equilibrium' % objName)
    plt.grid(True, zorder=0)
    coef = np.polyfit(Fe_line_regions['lower_state_eV'],x_over_h,1)
    poly1d_fn = np.poly1d(coef)
    x = np.linspace(.5, 5., 10)
    plt.plot(x, poly1d_fn(x), '-k', zorder=1)
#    plt.hlines(np.nanmean(x_over_h), 1, 5, zorder=0)
    plt.scatter(Fe_line_regions['lower_state_eV'][ind1], x_over_h[ind1], s=10, c='olive', zorder=2, label='Fe 1: %3i lines' % len(x_over_h[ind1]))
    plt.scatter(Fe_line_regions['lower_state_eV'][ind2], x_over_h[ind2], s=10, c='cyan', zorder=2, label='Fe 2: %3i lines' % len(x_over_h[ind2]))
    plt.legend()
    plt.savefig(ispec_dir+mySamOut_dir+objName+"_res_ExcitBal.pdf", bbox_inches='tight', pad_inches=0)
    plt.close()
    print('[Fe/H] by Fe 1: %2.2f; [Fe/H] by Fe 2: %2.2f' % (np.nanmean(x_over_h[Fe_line_regions['element']=='Fe 1']), np.nanmean(x_over_h[Fe_line_regions['element']=='Fe 2'])))
    
def EWabund(star_spectrum, star_continuum_model, linemasks, code='moog', mode='default'):
    global initial_teff, initial_logg, initial_MH, initial_vmic
    initial_alpha = ispec.determine_abundance_enchancements(initial_MH)
    max_iterations = 10
    #--- Metallicity grid limitation
    if initial_MH < -2.5:
        initial_MH = -2.5
    
    abundTargets = []
    for elem in linemasks['element']:
        if (elem not in abundTargets and elem != ''):
            abundTargets.append(elem)
    logging.info("HERE ARE OUR CONTESTANTS:" + str(abundTargets))
    
    model = ispec_dir + "input/atmospheres/ATLAS9.Castelli/" #MARCS.GES / ATLAS9.KuruczODFNEW
    # Load SPECTRUM abundances
    solar_abundances_file = ispec_dir + "input/abundances/Asplund.2009/stdatom.dat"
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)
    
    # Load model atmospheres
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)

    # Validate parameters
    if not ispec.valid_atmosphere_target(modeled_layers_pack, {'teff':initial_teff, 'logg':initial_logg, 'MH':initial_MH, 'alpha':initial_alpha}):
        msg = "The specified effective temperature, gravity (log g) and metallicity [M/H] \
                fall out of the atmospheric models."
        print(msg)

    ##--- EW abund -------------------------------------------------------------
    f = open(ispec_dir+mySamOut_dir+objName+"_res_Abund.txt", "w")
    f.write('Element\tN\t[X/H]\te[X/H]\t[X/Fe]\te[X/Fe]\tA(X)array\tA(X)+12.036array\t[X/H]array\n')
    for elem in abundTargets: #Calculate abundance for each element
        element_name = elem[:-2] # Fe 1 -> Fe, Sc 2 -> Sc, etc.
        logging.info("CALCULATING [" + element_name + "/H].")
        elem_line_regions = [line for line in linemasks if line[0]==elem]
        elem_line_regions = np.array(elem_line_regions)
        atmosphere_layers = ispec.interpolate_atmosphere_layers(modeled_layers_pack, {'teff':initial_teff, 'logg':initial_logg, 'MH':initial_MH, 'alpha':initial_alpha}, code=code)
        spec_abund, normal_abund, x_over_h, x_over_fe = ispec.determine_abundances(atmosphere_layers, \
            initial_teff, initial_logg, initial_MH, initial_alpha, elem_line_regions, solar_abundances, microturbulence_vel = initial_vmic, \
            verbose=1, code=code)
        ##--- Save results -----------------------------------------------------
        logging.info("Saving results...")
        f.write('%2s\t%i\t%1.2f\t%1.2f\t%1.2f\t%1.2f\t%s\t%s\n' % (elem, len(elem_line_regions), np.nanmean(x_over_h), np.nanstd(x_over_h), np.nanmean(x_over_fe), np.nanstd(x_over_fe), str(normal_abund), str(x_over_h)))
        print(elem+': ['+elem[:-2]+'/H] = '+str(x_over_h))
    f.close()
    
    if mode=='ssfl2l':
        ###--- SS L2L abund ----------------------------------------------------
        initial_vmac = ispec.estimate_vmac(initial_teff, initial_logg, initial_MH)
        initial_vsini = 1.6; initial_limb_darkening_coeff = 0.6; initial_vrad = 0.
        atomic_linelist_file = ispec_dir+mySamOut_dir+objName+"_LineList.txt"
        atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=np.min(star_spectrum['waveobs']), wave_top=np.max(star_spectrum['waveobs']))
        #atomic_linelist = atomic_linelist[atomic_linelist['theoretical_depth'] >= 0.01]
        chemical_elements_file = ispec_dir + "input/abundances/chemical_elements_symbols.dat"
        #chemical_elements_file = ispec_dir + "/input/abundances/molecular_symbols.dat"
        chemical_elements = ispec.read_chemical_elements(chemical_elements_file)
        isotope_file = ispec_dir + "input/isotopes/SPECTRUM.lst"
        isotopes = ispec.read_isotope_data(isotope_file)
        linelist_free_loggf = None
        free_params = []
        f = open(ispec_dir+mySamOut_dir+objName+"_res_L2LSynthAbund.txt","w")
        f.write('Element\twl (nm)\tTurbocode\t[X/H]\te[X/H]\t[X/Fe]\te[X/Fe]\n')
        for i, line in enumerate(linemasks): #Calculate abundance for each line
            if line['element'][-1] == ' ' or int(line['element'][-1]) > 2:
                continue
            #if elem[:-2] == 'CO':
            #    continue
            element_name = line['element'][:-2] # Fe 1 -> Fe, Sc 2 -> Sc, etc.
            free_abundances = ispec.create_free_abundances_structure([element_name], chemical_elements, solar_abundances)
            free_abundances['Abund'] += initial_MH # Scale to metallicity
            logging.info("CALCULATING SYNTHETIC [" + element_name + "/Fe] at " + str(np.round(line['wave_peak'],3)) + " nm (line #" + str(i+1) + "/"+str(len(linemasks))+").")
            individual_line_regions = linemasks[i:i+1]
            segments = ispec.create_segments_around_lines(individual_line_regions, margin=0.25)
            wfilter = ispec.create_wavelength_filter(star_spectrum, regions=segments) # Only use the segment
            if len(star_spectrum[wfilter]) == 0 or np.any(star_spectrum['flux'][wfilter] == 0):
                continue
            obs_spec, modeled_synth_spectrum, params, errors, abundances_found, loggf_found, status, stats_linemasks = \
                    ispec.model_spectrum(star_spectrum[wfilter], star_continuum_model, \
                    modeled_layers_pack, atomic_linelist, isotopes, solar_abundances, free_abundances, linelist_free_loggf, initial_teff, \
                    initial_logg, initial_MH, initial_alpha, initial_vmic, initial_vmac, initial_vsini, \
                    initial_limb_darkening_coeff, initial_R, initial_vrad, free_params, segments=segments, \
                    linemasks=individual_line_regions, \
                    enhance_abundances=True, \
                    use_errors = False, \
                    vmic_from_empirical_relation = False, \
                    vmac_from_empirical_relation = False, \
                    max_iterations=max_iterations, \
                    tmp_dir = None, \
                    code=code)
            logging.info("SAVING SYNTHETIC [" + element_name + "/Fe] = " + str(np.round(abundances_found[0][5],2)) + " at " + str(np.round(line['wave_peak'],3)) + " nm (line #" + str(i+1) + "/"+str(len(linemasks))+").")
            if abundances_found[0][7] > 5.: #Bad uncertainties
                abundances_found[0][7] = -1.
                abundances_found[0][9] = -1.
            f.write('%2s\t%4.3f\t%s\t%1.2f\t%1.2f\t%1.2f\t%1.2f\n' % (abundances_found[0][2], line['wave_peak'], line['turbospectrum_species'], abundances_found[0][3], abundances_found[0][7], abundances_found[0][5], abundances_found[0][9]))
            ##--- Save synthetic spectrum --------------------------------------
            synthSpecName = ispec_dir+mySamOut_dir+"L2LSynthSpec/"
            if not os.path.exists(synthSpecName):
                os.makedirs(synthSpecName)
            fOut = open(synthSpecName+objName+"_"+ element_name+"_"+ str(np.round(line['wave_peak'],2))+"nm.txt", "w")
            for s in modeled_synth_spectrum:
                fOut.write('%9s\t%9s\t%9s\n' % (np.round(s['waveobs'],5),np.round(s['flux'],7), np.round(s['err'],7)))
            fOut.close()
        f.close()

def SaveSpec(star_spectrum, spectrum_type):
    f = open(ispec_dir+mySamOut_dir+objName+"_"+spectrum_type+".txt", "w")
    f.write('waveobs\tflux\terr\n')
    for s in star_spectrum:
        f.write('%9s\t%9s\t%9s\n' % (np.round(s['waveobs'],5),np.round(s['flux'],7), np.round(s['err'],7)))
    f.close()



def StepReduc(objName):
    visits = {
        #"DFCyg": '00972481'
    }
    star_spectrum = ispec.read_spectrum(ispec_dir+mySamIn_dir+objName+".txt")
    #star_spectrum = ContFitAndNorm(star_spectrum)
    star_continuum_model = ispec.fit_continuum(star_spectrum, fixed_value=1.0, model="Fixed value")
    star_spectrum = CosmicFilter(star_spectrum, star_continuum_model)
    star_spectrum, rv, rv_err = RVCorr(star_spectrum)
#    star_spectrum, snr = SNRErrCalc(star_spectrum)
    SaveSpec(star_spectrum, "CleanSpec")
    #ListCreation(np.min(star_spectrum['waveobs']), np.max(star_spectrum['waveobs']))
    return(star_spectrum, star_continuum_model, rv, rv_err)

def StepFind(star_spectrum, star_continuum_model, rv): # A manual filtering should be done after this step
    linemasks = LineFit(star_spectrum, star_continuum_model, rv, mode="seek")
    return(linemasks)

def StepFilter(star_spectrum, star_continuum_model, rv):
    linemasks = LineFit(star_spectrum, star_continuum_model, rv, mode="tweak")
    #EWparam(star_spectrum, star_continuum_model, linemasks, code="moog")
    #EWabund(star_spectrum, star_continuum_model, linemasks, code="moog")
    return(linemasks)

def StepStud(star_spectrum, star_continuum_model, rv):
    linemasks = LineFit(star_spectrum, star_continuum_model, rv, mode="pick")
    #EWparam(star_spectrum, star_continuum_model, linemasks, code="moog")
    #EWabund(star_spectrum, star_continuum_model, linemasks, code="moog") #, mode="ssfl2l"
    return(linemasks)

def synthesize_spectrum(code="turbospectrum"): # for Aashique
    #--- Synthesizing spectrum -----------------------------------------------------
    # Parameters
    global initial_teff, initial_logg, initial_MH, initial_R, initial_vrad
    teff = 3500. #initial_teff
    logg = 0.5 #initial_logg
    MH = -0.5 #initial_MH
    alpha = ispec.determine_abundance_enchancements(initial_MH)
    microturbulence_vel = 1.07
    macroturbulence = 4.21
    vsini = 1.6
    limb_darkening_coeff = 0.6
    resolution = 50000.
    wave_step = 0.01

    # Wavelengths to synthesis
    #regions = ispec.read_segment_regions(ispec_dir + "/input/regions/fe_lines_segments.txt")
    regions = None
    wave_base = 1500.0
    wave_top = 1700.0


    # Selected model amtosphere, linelist and solar abundances
    model = ispec_dir + "/input/atmospheres/MARCS.GES/"

    # Load chemical information and linelist
    atomic_linelist_file = ispec_dir + "/input/linelists/transitions/APOGEE.Thomas/excALLibur.tsv"
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=wave_base, wave_top=wave_top)
    #atomic_linelist = atomic_linelist[atomic_linelist['theoretical_depth'] >= 0.01] # Select lines that have some minimal contribution in the sun
    #atomic_linelist = atomic_linelist[atomic_linelist['element'] != 'CN 1'] # Select lines that have some minimal contribution in the sun
    #atomic_linelist = atomic_linelist[atomic_linelist['element'] != 'N 1']

    isotope_file = ispec_dir + "/input/isotopes/SPECTRUM.lst"
    isotopes = ispec.read_isotope_data(isotope_file)
    #ratio = 3300.
    #isotopes[10][3]=ratio/(ratio+1.); isotopes[11][3] = 1./(ratio+1.) #13C
    #isotopes[12][3]=ratio/(ratio+1.); isotopes[13][3] = 1./(ratio+1.) #15N
    #isotopes[15][3]=(1.-isotopes[16][3])/(ratio+1.); isotopes[14][3] = 1.-isotopes[15][3]-isotopes[16][3] #17O
    #isotopes[16][3]=(1.-isotopes[15][3])/(ratio+1.); isotopes[14][3] = 1.-isotopes[16][3]-isotopes[15][3] #18O

    # Load model atmospheres
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)
    # Load SPECTRUM abundances
    solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2009/stdatom.dat"
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)

    ## Custom fixed abundances
    chemical_elements = ispec.read_chemical_elements(ispec_dir + "/input/abundances/chemical_elements_symbols.dat")
    fixed_abundances = ispec.create_free_abundances_structure(["C", "N", "O"], chemical_elements, solar_abundances)
    fixed_abundances['Abund'] = [8.93-12.04, 8.33-12.04, 9.19-12.04] # Abundances in SPECTRUM scale (i.e., x - 12.0 - 0.036) and in the same order ["C", "N", "O"]
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
    f = open(ispec_dir+"AnibalTest/test.txt", "w") #"test_synth_4000K_AGBtip_O18_3300.txt"
    f.write('waveobs\tflux\terr\n')
    for s in synth_spectrum:
        f.write('%.5f\t%.7f\t%.7f\n' % (s['waveobs'], s['flux'], s['err']))
    f.close()

def precompute_synthetic_grid(code="turbospectrum"):
    global initial_teff, initial_logg, initial_MH, initial_R, initial_vmic
    precomputed_grid_dir = ispec_dir + "example_grid/"

    ranges = np.recarray((2,),  dtype=[('teff', int), ('logg', float), ('MH', float), ('alpha', float), ('vmic', float)])
    ranges['teff'][0] = initial_teff
    ranges['logg'][0] = initial_logg
    ranges['MH'][0] = initial_MH
    ranges['alpha'][0] = 0.0
    ranges['vmic'][0] = initial_vmic
    ranges['teff'][1] = initial_teff
    ranges['logg'][1] = initial_logg
    ranges['MH'][1] = initial_MH
    ranges['alpha'][1] = 0.0
    ranges['vmic'][1] = initial_vmic

    # Wavelengths
    initial_wave = 300.0
    final_wave = 1100.0
    step_wave = 0.01
    wavelengths = np.arange(initial_wave, final_wave, step_wave)

    to_resolution = 50000 # Individual files will not be convolved but the grid will be (for fast comparison)
    number_of_processes = 1 # It can be parallelized for computers with multiple processors

    # Selected model amtosphere, linelist and solar abundances
    model = ispec_dir + "/input/atmospheres/ATLAS9.Castelli/"
    atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.300_1100nm/atomic_lines.tsv"
    isotope_file = ispec_dir + "/input/isotopes/SPECTRUM.lst"

    # Load chemical information and linelist
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=initial_wave, wave_top=final_wave)
    atomic_linelist = atomic_linelist[atomic_linelist['theoretical_depth'] >= 0.01] # Select lines that have some minimal contribution in the sun
    isotopes = ispec.read_isotope_data(isotope_file)
    solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2009/stdatom.dat"

    # Load model atmospheres
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)
    # Load SPECTRUM abundances
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)

    fixed_abundances = None

    ispec.precompute_synthetic_grid(precomputed_grid_dir, ranges, wavelengths, to_resolution, \
                                    modeled_layers_pack, atomic_linelist, isotopes, solar_abundances, \
                                    segments=None, number_of_processes=number_of_processes, \
                                    code=code, steps=False)

def interpolate_atmosphere(code="turbospectrum"):
    global initial_teff, initial_logg, initial_MH, initial_R, initial_vmic
    initial_alpha = ispec.determine_abundance_enchancements(initial_MH)
    model = ispec_dir + "/input/atmospheres/ATLAS9.Castelli/"
    # Load model atmospheres
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)

    # Validate parameters
    if not ispec.valid_atmosphere_target(modeled_layers_pack, {'teff':initial_teff, 'logg':initial_logg, 'MH':initial_MH, 'alpha':initial_alpha}):
        msg = "The specified effective temperature, gravity (log g) and metallicity [M/H] \
                fall out of theatmospheric models."
        print(msg)

    # Prepare atmosphere model
    atmosphere_layers = ispec.interpolate_atmosphere_layers(modeled_layers_pack, {'teff':initial_teff, 'logg':initial_logg, 'MH':initial_MH, 'alpha':initial_alpha}, code=code)
    atmosphere_layers_file = ispec_dir + "example_grid/%s.txt" % (objName)
    atmosphere_layers_file = ispec.write_atmosphere(atmosphere_layers, initial_teff, initial_logg, initial_MH, atmosphere_filename=atmosphere_layers_file, code=code)

if __name__ == '__main__':
    star_spectrum, star_continuum_model, rv, rv_err = StepReduc(objName)
    #linemasks = StepFind(star_spectrum, star_continuum_model, rv)
    #linemasks = StepFilter(star_spectrum, star_continuum_model, rv)
    linemasks = StepStud(star_spectrum, star_continuum_model, rv)
    #synthesize_spectrum(code="turbospectrum")
    precompute_synthetic_grid()
    interpolate_atmosphere()
    pass
