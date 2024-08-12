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
#initial_teff = 7530.0; initial_logg = 0.93; initial_MH = -0.7 #J003643
#initial_teff = 7458.0; initial_logg = 0.5; initial_MH = -0.63 #J003643 low resolution values
#initial_teff = 6000.0; initial_logg = 0.5; initial_MH = -1.5 #J005107
#initial_teff = 6000.0; initial_logg = 0.5; initial_MH = -1.56 #J005107 old
#initial_teff = 7000.0; initial_logg = 1.25; initial_MH = -0.64 #HD158616
#initial_teff = 4900.0; initial_logg = 0.0; initial_MH = -1.5 #MACHO47
#initial_teff = 8250; initial_logg = 1.0; initial_MH = -1.18 #J005252
#initial_teff = 5771; initial_logg = 4.44; initial_MH = 0 #Sun
initial_teff = 8000; initial_logg = 2.5; initial_MH = 0.5 
#initial_teff = 6000.0; initial_logg = 1.0; initial_MH = -1.5
initial_vmic = 2; initial_R = 80000
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
    logging.info("MEDIAN WAVE RANGE =" + str(np.median(star_spectrum['waveobs'][1:]-star_spectrum['waveobs'][:-1])))
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
    #--- Normalize ----------------------------------------------------------
    #star_spectrum = ispec.normalize_spectrum(star_spectrum, star_continuum_model, consider_continuum_errors=False)
    star_continuum_model = ispec.fit_continuum(star_spectrum, fixed_value=1.0, model="Fixed value")
    return(star_spectrum, star_continuum_model)
    
def RVCorr(star_spectrum):
    #--- Radial Velocity determination with template -------------------------
    logging.info("Radial velocity determination with mask...")
      # - Read synthetic template
    mask_file = ispec_dir + "input/linelists/CCF/VALD.Sun.300_1100nm/mask.lst"
    ccf_mask = ispec.read_cross_correlation_mask(mask_file)

    models, ccf = ispec.cross_correlate_with_mask(star_spectrum, ccf_mask, \
                            lower_velocity_limit=-300, upper_velocity_limit=300, \
                            velocity_step=1.0, mask_depth=0.01, \
                            fourier=False)
    
    # Number of models represent the number of components
    components = len(models)
    # First component:
    rv = np.round(models[0].mu(), 2) # km/s
    rv_err = np.round(models[0].emu(), 2) # km/s
    # - Read synthetic template
    #smooth_R = 50000.
    #smooth_spectrum = ispec.convolve_spectrum(star_spectrum, smooth_R, from_resolution=initial_R)
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Arcturus.372_926nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Sun.372_926nm/template.txt.gz")
    #logging.info("Radial velocity determination with Arcturus template...")
    #models, ccf = ispec.cross_correlate_with_template(star_spectrum, template,\
    #        lower_velocity_limit=0., upper_velocity_limit=120.,\
    #        velocity_step=0.25,fourier=True)
    # Plotting CCF (to exclude those contaminated by the shocks)
    #vel = [c[0] for c in ccf] # Velocity
    #val = [c[1] for c in ccf] # CCF values
    #from matplotlib import pyplot as plt
    #plt.rcParams["font.family"] = "Times New Roman"
    #plt.xlabel('Velocity (km/s)')
    #plt.ylabel('Normalised CCF')
    #plt.title(objName)
    #plt.plot(vel, val)
    #plt.grid()
    #plt.show()
    # Number of models represent the number of components
    #components = len(models)
    # First component:
    #rv = np.round(models[0].mu(), 2) # km/s
    #rv_err = np.round(models[0].emu(), 2) # km/s
    
    #--- Radial Velocity correction ------------------------------------------
    logging.info("Radial velocity correction... %.2f +/- %.2f" % (rv, rv_err))
    star_spectrum = ispec.correct_velocity(star_spectrum, rv)
    return(star_spectrum, rv, rv_err)
    
def CosmicFilter(spec, star_continuum_model):
    # Spectrum should be already normalized
    step = spec['waveobs'][len(spec)//2+1] - spec['waveobs'][len(spec)//2]
    cosmics = ispec.create_filter_cosmic_rays(spec,star_continuum_model,\
              resampling_wave_step=step, window_size=15, \
              variation_limit=0.5)
    spec = spec[~cosmics]
    return(spec)

def SNRErrCalc(star_spectrum):
    #--- Estimate SNR from flux and errors from SNR --------------------------
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
    
def ListCreation():
    #--- Calculate theoretical equivalent widths and depths for a linelist --
    Aname, Ateff, Alogg, Amet, Avmic, Acomm = np.loadtxt(ispec_dir + "Spoiler.txt", delimiter='\t', dtype=np.dtype([('name','U8'), ('teff',np.float64), ('logg',np.float64), ('met',np.float64), ('vmic',np.float64), ('comm','U5')]), skiprows=1, unpack=True)
    index = (Aname==objName)
    teff = float(Ateff[index])
    logg = float(Alogg[index])
    met = float(Amet[index])
    vmic = float(Avmic[index])
    logging.info("CREATING A LINE LIST FOR THE FOLLOWING AP: %4.0f, %1.1f, %1.2f, %1.1f" % (teff, logg, met, vmic))
    alpha = ispec.determine_abundance_enchancements(met)
    model = ispec_dir + "/input/atmospheres/ATLAS9.KuruczODFNEW/"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.KuruczODFNEW/"
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)
    atmosphere_layers = ispec.interpolate_atmosphere_layers(modeled_layers_pack, {'teff':teff, 'logg':logg, 'MH':met, 'alpha':alpha})
    isotopes = ispec.read_isotope_data(ispec_dir + "/input/isotopes/SPECTRUM.lst")
    solar_abundances = ispec.read_solar_abundances(ispec_dir + "/input/abundances/Asplund.2009/stdatom.dat")
    atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.300_1100nm/atomic_lines.tsv"
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=np.min(star_spectrum['waveobs']), wave_top=np.max(star_spectrum['waveobs']))
    new_atomic_linelist = ispec.calculate_theoretical_ew_and_depth(atmosphere_layers, \
            teff, logg, met, alpha, \
            atomic_linelist, isotopes, solar_abundances, microturbulence_vel=vmic, \
            verbose=1, gui_queue=None, timeout=900)
    ispec.write_atomic_linelist(new_atomic_linelist, linelist_filename=ispec_dir + mySamOut_dir + objName + "_LineList.txt")
    
def LineFit(star_spectrum, star_continuum_model, use_ares):
    #--- Reading required files ----------------------------------------------
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
    
    #smooth_R = 50000.
    #smooth_spectrum = ispec.convolve_spectrum(star_spectrum, smooth_R, from_resolution=initial_R)
    
    # Line regions
    #line_regions = ispec.read_line_regions(ispec_dir + mySamIn_dir + "SZMon_J051845.txt"); ID = 'note'
    #line_regions = ispec.read_line_regions(ispec_dir + mySamIn_dir + "SZMon_Meghna.txt"); ID = 'note'
    #line_regions = ispec.read_line_regions(ispec_dir + mySamIn_dir + "SZMon_gen.txt"); ID = 'note'
    #line_regions = ispec.read_line_regions(ispec_dir + mySamIn_dir + "DFCyg_gen.txt"); ID = 'note'
    #line_regions = ispec.read_line_regions(ispec_dir + mySamIn_dir + "J005252_gen.txt"); ID = 'note'
    #line_regions = ispec.read_line_regions(ispec_dir + mySamIn_dir + "J005107_gen_ob3.txt"); ID = 'note'
    line_regions = ispec.read_line_regions(ispec_dir + mySamIn_dir + "FinalLinelist_%7s.txt" % objName); ID = 'note'

    #####--- Find linemasks -----------------------------------------------------
    #line_regions = ispec.find_linemasks(star_spectrum, star_continuum_model,\
    #                        atomic_linelist=atomic_linelist, \
    #                       max_atomic_wave_diff = 0.005, \
    #                        telluric_linelist=None, \
    #                        vel_telluric=None, \
    #                        minimum_depth=0.1, maximum_depth=0.5, \
    #                        smoothed_spectrum=None, \
    #                        check_derivatives=False, \
    #                        discard_gaussian=False, discard_voigt=True, \
    #                        closest_match=False)
    ## Discard bad masks
    #flux_peak = star_spectrum['flux'][line_regions['peak']]
    #flux_base = star_spectrum['flux'][line_regions['base']]
    #flux_top = star_spectrum['flux'][line_regions['top']]
    #bad_mask = np.logical_or(line_regions['wave_peak'] <= line_regions['wave_base'], line_regions['wave_peak'] >= line_regions['wave_top'])
    #bad_mask = np.logical_or(bad_mask, flux_peak >= flux_base)
    #bad_mask = np.logical_or(bad_mask, flux_peak >= flux_top)
    #bad_mask = np.logical_or(bad_mask, line_regions['element']=='')
    #bad_mask = np.logical_or(bad_mask, line_regions['molecule']!='F')
    #bad_mask = np.logical_or(bad_mask, int(line_regions['element'][-1])>2)
    #line_regions = line_regions[~bad_mask]
    #iron_mask = np.logical_or(line_regions['element']=='Fe 1', line_regions['element']=='Fe 2')
    #line_regions = line_regions[iron_mask]
    #ID = 'element'
    #f = open(ispec_dir+mySamOut_dir+objName+"_F.txt", "w")
    #f.write('wave_peak\twave_base\twave_top\tnote\n')
    #for l in line_regions:
    #    f.write('%.4f\t%.4f\t%.4f\t%s\n' % (l['wave_peak'], l['wave_base'], l['wave_top'], l['element']))
    #f.close()
    ##############################################################################
    ids = [l[ID] for l in line_regions]
    ids_counted = {item:ids.count(item) for item in ids}
    logging.info("LINES READ: " + str(ids_counted))
    logging.info("TOTAL NUMBER: " + str(len(line_regions)))
    ##--- Fit lines ----------------------------------------------------------
    logging.info("Fitting lines...")
    print(len(line_regions)) #9598
    line_regions = ispec.adjust_linemasks(star_spectrum, line_regions, max_margin=0.5)
    print(len(line_regions))
    linemasks = ispec.fit_lines(line_regions, star_spectrum, star_continuum_model, \
                atomic_linelist = atomic_linelist, \
                max_atomic_wave_diff = 0.005, \
                telluric_linelist = None, \
                smoothed_spectrum = None, \
                check_derivatives = False, \
                vel_telluric = None, discard_gaussian=False, \
                discard_voigt=True, \
                free_mu=True, crossmatch_with_mu=False, closest_match=False)
    print(len(linemasks)) 
    #print(linemasks)
    line_regions.sort(order='wave_peak')
    linemasks.sort(order='wave_peak')
           
    #############################################################################    
    f = open(ispec_dir+mySamOut_dir+objName+"_lineregs.txt", "w")
    for l in linemasks:
        f.write('%8s\t%8s\t%8s\t%5s\n' % (np.round(l['wave_peak'],4), np.round(l['wave_base'],4), np.round(l['wave_top'],4), l['element']))
    f.close()
    #############################################################################      
    ispec.write_line_regions(linemasks, ispec_dir+mySamOut_dir+objName+ "_LineFit.txt", extended=True)
    f = open(ispec_dir+mySamOut_dir+objName+"_LineFit_short1.txt", "w")
    for l in line_regions:
        f.write('%5s\t%8s\n' % (l[ID], np.round(l['wave_peak'],4)))
    f.close()
    
    # Discard lines that are not cross matched with the same original element stored in the note. Uncomment the below line if the linelist is not manually cleaned up.
    linemasks = linemasks[linemasks['element'] == line_regions[ID]]
    print("After element=note")
    print(len(linemasks)) 
    #############################################################################    
    f = open(ispec_dir+mySamOut_dir+objName+"_linemask_AfterElementNoteComparison", "w")
    for l in linemasks:
        f.write('%8s\t%8s\t%8s\t%5s\n' % (np.round(l['wave_peak'],4), np.round(l['wave_base'],4), np.round(l['wave_top'],4), l['element']))
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
        f.write('%8s\t%8s\t%8s\t%5s\n' % (np.round(l['wave_peak'],4), np.round(l['wave_base'],4), np.round(l['ew'],2), l['element']))
    f.close()
    #############################################################################  
    linemasks = linemasks[linemasks['ew']>5.]#4 #5
    linemasks = linemasks[linemasks['ew']<250.]#250
    linemasks.sort(order='turbospectrum_species')
    #############################################################################    
    f = open(ispec_dir+mySamOut_dir+objName+"_linemask_AfterEWFilter", "w")
    for l in linemasks:
        f.write('%8s\t%8s\t%8s\t%5s\n' % (np.round(l['wave_peak'],4), np.round(l['wave_base'],4), np.round(l['wave_top'],4), l['element']))
    f.close()
    #############################################################################     
    ispec.write_line_regions(linemasks, ispec_dir+mySamOut_dir+objName+ "_LineFit.txt", extended=True)
    f = open(ispec_dir+mySamOut_dir+objName+"_LineFit_short.txt", "w")
    for l in linemasks:
        f.write('%5s\t%8s\t%8s\t%8s\t%8s\t%8s\n' % (l['element'], np.round(l['wave_nm'],4),np.round(l['ew'],2), np.round(l['ewr'],2), np.round(l['lower_state_eV'],2),np.round(l['rms'],2)))
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
        f.write('%8s\t%8s\t%8s\t%5s\n' % (np.round(l['wave_peak'],4), np.round(l['wave_base'],4), np.round(l['ew'],2), l['element']))
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
    
    if use_ares:
        # Replace the measured equivalent widths by the ones computed by ARES
        old_linemasks = linemasks.copy()
        ### Different rejection parameters (check ARES papers):
        ##   - http://adsabs.harvard.edu/abs/2007A%26A...469..783S
        ##   - http://adsabs.harvard.edu/abs/2015A%26A...577A..67S
        #linemasks = ispec.update_ew_with_ares(normalized_star_spectrum, linemasks, rejt="0.995", tmp_dir=None, verbose=0)
        #linemasks = ispec.update_ew_with_ares(normalized_star_spectrum, linemasks, rejt="3;5764,5766,6047,6052,6068,6076", tmp_dir=None, verbose=0)
        snr = 80#np.median(linemasks['snr'])#50
        linemasks = ispec.update_ew_with_ares(star_spectrum, linemasks, rejt="%s" % (snr), tmp_dir=None, verbose=0)
    
    #LineComparPlot(linemasks)
    LineFitPlot(star_spectrum, linemasks)
    return(linemasks)
    
def LineFitPlot(star_spectrum, linemasks):
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = PdfPages(ispec_dir+mySamOut_dir+objName+"_FittedLines.pdf")
    for i in range(len(linemasks)):
        logging.info('PLOTTING LINE ' + str(i+1) + '/' + str(len(linemasks)))
        numGap = 500 #100
        spec = star_spectrum[linemasks['peak'][i]-numGap:linemasks['peak'][i]+numGap]
        gauss = 1.+linemasks['A'][i]*np.exp(-(spec['waveobs']-linemasks['mu'][i])**2/(2.*linemasks['sig'][i]**2))
        if len(spec['waveobs'])<numGap:
            continue
        step = spec['waveobs'][numGap]-spec['waveobs'][numGap-1]
        from_x = numGap - int(6.*linemasks['sig'][i]/step)
        to_x = numGap + int(6.*linemasks['sig'][i]/step)
        wave_filter_custom = (spec['waveobs'] >= linemasks['wave_base'][i]) & (spec['waveobs'] <= linemasks['wave_top'][i])
        line = spec[wave_filter_custom]
        lineEndsX = [line['waveobs'][0], line['waveobs'][-1]]
##        lineEndsX = [linemasks['wave_base'][i]-linemasks['wave_peak'][i]+100, linemasks['wave_top'][i]-linemasks['wave_peak'][i]+100]
#        lineEndsY = [line['flux'][0], line['flux'][-1]]
        lineEndsY = [1., 1.]
        #cont_a = (line['flux'][-1]-line['flux'][0])/(line['waveobs'][-1]-line['waveobs'][0])
        #cont_b = (line['waveobs'][-1]*line['flux'][0]-line['waveobs'][0]*line['flux'][-1])/(line['waveobs'][-1]-line['waveobs'][0])
        cont_a = 0.; cont_b = 1.
        continuum_custom = cont_a*spec['waveobs'] + cont_b
        
        fig = plt.figure(figsize=(11,8))
        ax = fig.add_subplot(1, 1, 1)
        plt.xlim([spec['waveobs'][0],spec['waveobs'][-1]])
        plt.ylim([0.2,1.2])
        plt.xlabel('$\lambda$ (nm)')
        plt.title(linemasks['element'][i] + ' (%i/%i), EW = %3.1f mA, initial guess EW = %3.1f mA' % (i+1,len(linemasks),linemasks['ew'][i], linemasks['theoretical_ew'][i]))
        ax.plot(spec['waveobs'], spec['flux'], '-k')
        #ax.plot(lineEndsX, lineEndsY, 'co', ms=5)
        #ax.plot(lineEndsX, lineEndsY, 'c-', lw=3)
        ax.plot(spec['waveobs'][from_x:to_x], gauss[from_x:to_x], '-r')
#        ax.axvline(linemasks['wave_base'][i], c='yellow', zorder=0)
#        ax.axvline(linemasks['wave_top'][i], c='yellow', zorder=0)
        ax.axvline(linemasks['wave_peak'][i], c='red', zorder=1)
        #ax.axvline(lineEndsX[0], c='olive', ls=':', zorder=1)
        #ax.axvline(lineEndsX[-1], c='olive', ls=':', zorder=1)
        ax.axhline(1., c='gray', ls='--', zorder=0)
        ax.fill_between(lineEndsX, y1=0., y2=1., color='none', edgecolor='olive', hatch='\\\\\\', zorder=0)
        ax.fill_between(spec['waveobs'][from_x:to_x], gauss[from_x:to_x], continuum_custom[from_x:to_x], color='none', edgecolor='cyan', hatch='///', zorder=0)
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
    
def EWparam(star_spectrum, star_continuum_model, linemasks, code='moog'):
    #--- Model spectra from EW --------------------------------------------------
    global initial_teff, initial_logg, initial_MH, initial_vmic
    initial_alpha = ispec.determine_abundance_enchancements(initial_MH)
    max_iterations = 10

    #--- Model spectra -------------------------------------------------------
    logging.info("Last initializations...")
    # Selected model amtosphere, linelist and solar abundances
    model = ispec_dir + "input/atmospheres/MARCS.GES/"
    #model = ispec_dir + "input/atmospheres/ATLAS9.Castelli/"

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
    f.write('wave_peak\twave_base\twave_top\tnote\n')
    for l in linemasks:
        f.write('%.4f\t%.4f\t%.4f\t%s\n' % (l['wave_peak'], l['wave_base'], l['wave_top'], l['element']))
    f.close()
#############################################################################    
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
                        code=code)
    params, errors, status, x_over_h, selected_x_over_h, fitted_lines_params, used_linemasks = results
    #print(len(linemasks), len(x_over_h), len(used_linemasks))
    ##--- Plotting the constraints' plots ------------------------------------
    CnstrPlot(linemasks, x_over_h)
    
    ##--- Save results -------------------------------------------------------
    logging.info("Saving results...")
    f = open(ispec_dir+mySamOut_dir+objName+"_res_StePar.txt", "w")
    f.write('Teff\teTeff\tlogg\telogg\tMH\teMH\tvmic\tevmic\n')
    f.write('%4.0f\t%4.0f\t%1.2f\t%1.2f\t%1.2f\t%1.2f\t%1.2f\t%1.2f\n' % (params['teff'], errors['teff'], params['logg'], errors['logg'], params['MH'], errors['MH'], params['vmic'], errors['vmic']))
    f.close()
    initial_teff = params['teff']
    initial_logg = params['logg']
    initial_MH = params['MH']
    initial_vmic = params['vmic']
    initial_alpha = ispec.determine_abundance_enchancements(initial_MH)

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
    f.write('wave_peak\twave_base\twave_top\tlower_stste_ev\tewr\ttheoretical_ew\tnote\n')
    for l in Fe_line_regions:
        f.write('%.4f\t%.4f\t%.4f\t%.4f\t%.2f\t%.2f\t%s\n' % (l['wave_peak'], l['wave_base'], l['wave_top'], l['lower_state_eV'], l['ewr'], l['theoretical_ew'], l['element']))
    f.close()
    for i in range(len(x_over_h)):
        print('%s: CWL=%.5f, [Fe/H]=%.2f, EW=%.2f, thEW=%.2f, rEW=%.2f, eV=%.2f' % (Fe_line_regions['element'][i], Fe_line_regions['wave_peak'][i], x_over_h[i], Fe_line_regions['ew'][i], Fe_line_regions['theoretical_ew'][i], Fe_line_regions['ewr'][i], Fe_line_regions['lower_state_eV'][i]))
    
    f = open(ispec_dir+mySamOut_dir+objName+"_FinalFeLinelist", "w")
    f.write('wave_peak\twave_base\twave_top\tnote\n')
    for l in Fe_line_regions:
        f.write('%.4f\t%.4f\t%.4f\t%s\n' % (l['wave_peak'], l['wave_base'], l['wave_top'], l['element']))
    f.close()
    
    
    ## v_mic correction
    plt.xlim([-6.,-4.2])
    plt.ylim([np.nanmean(x_over_h)-1,np.nanmean(x_over_h)+1])
    plt.xlabel("reduced EW")
    plt.ylabel("[Fe/H]")
    plt.title('%7s, $v_{mic}$ correction' % objName)
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
    plt.title('%7s, excitation equilibrium' % objName)
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
    
def EWabund(star_spectrum, star_continuum_model, linemasks, code):
    global initial_teff, initial_logg, initial_MH, initial_vmic
    initial_alpha = ispec.determine_abundance_enchancements(initial_MH)
    max_iterations = 10
    #--- Metallicity grid limitation
    if initial_MH < -2.5:
        initial_MH = -2.5
    
    abundTargets = []
    abundances=[]
    for elem in linemasks['element']:
        if (elem not in abundTargets and elem != ''):
            abundTargets.append(elem)
    logging.info("HERE ARE OUR CONTESTANTS:" + str(abundTargets))
    
    #for i, row in enumerate(linemasks):
    #        if row['element'] == "N 1":
    #            row['ew']=10
    
    #model = ispec_dir + "input/atmospheres/ATLAS9.Castelli/"
    model = ispec_dir + "input/atmospheres/MARCS.GES/"
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
    initial_vmic=3.25 #change initial values of micro turbulent velocity here to estimate error for abundances
    ##--- EW abund -----------------------------------------------------------
    m = open(ispec_dir+mySamOut_dir+objName+"_res_Epsilon_EW.txt", "w")
    m.write('Z\tElement\twavelength_A\tEquivalent width\tlog epsilon\n')
    for l in linemasks: 
        m.write('%s\t%s\t%.4f\t%1.2f\n' % (l['turbospectrum_species'],l['element'],l['wave_peak'],l['ew']))
      
    f = open(ispec_dir+mySamOut_dir+objName+"_res_Abund.txt", "w")
    f.write('Element\tN\tZ\t[X/H]\tlog(epsilon)\t[X/Fe]\te[X/Fe]\n')
    for elem in abundTargets: #Calculate abundance for each element
        element_name = elem[:-2] # Fe 1 -> Fe, Sc 2 -> Sc, etc.
        logging.info("CALCULATING [" + element_name + "/Fe].")
        elem_line_regions = [line for line in linemasks if line[0]==elem]
        elem_line_regions = np.array(elem_line_regions)
        atmosphere_layers = ispec.interpolate_atmosphere_layers(modeled_layers_pack, {'teff':initial_teff, 'logg':initial_logg, 'MH':initial_MH, 'alpha':initial_alpha}, code=code) #change initial values of atmospheric values here to estimate error for abundances
        spec_abund, normal_abund, x_over_h, x_over_fe = ispec.determine_abundances(atmosphere_layers, \
            initial_teff, initial_logg, initial_MH, initial_alpha, elem_line_regions, solar_abundances, microturbulence_vel = initial_vmic, \
            verbose=1, code=code)
        #print(element_name, x_over_h, spec_abund+12.036, x_over_fe)
        print(element_name, spec_abund+12.036)
        abundances.append(normal_abund)
        ##--- Save results --------------------------------------------------
        logging.info("Saving results...")
        #f.write('%2s\t%i\t%1.2f\t%1.2f\t%1.2f\n' % (elem, len(elem_line_regions), normal_abund, np.mean(x_over_fe), np.nanstd(x_over_fe))
        #f.write('%2s\t%i\t%1.2f\t%1.2f\t%1.2f\t%1.4f\n' % (elem, len(elem_line_regions), normal_abund, 0.00, np.mean(x_over_fe), stdev(x_over_fe)))
        f.write('%2s\t%i\t%2f\t%1.2f\t%1.2f\t%1.2f\t%1.2f\n' % (elem, len(elem_line_regions), 0, np.mean(x_over_h), np.mean(normal_abund), np.mean(x_over_fe), np.nanstd(x_over_fe)))
       
    f.close()
    m.close()
   
    ##--- SS L2L abund --------------------------------------------------------
    #initial_vmac = ispec.estimate_vmac(initial_teff, initial_logg, initial_MH)
    #initial_vsini = 1.6; initial_limb_darkening_coeff = 0.6; initial_vrad = 0.
    #atomic_linelist_file = ispec_dir+mySamOut_dir+objName+"_LineList.txt"
    #atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=np.min(star_spectrum['waveobs']), wave_top=np.max(star_spectrum['waveobs']))
    #atomic_linelist = atomic_linelist[atomic_linelist['theoretical_depth'] >= 0.01]
    #chemical_elements_file = ispec_dir + "input/abundances/chemical_elements_symbols.dat"
    ##chemical_elements_file = ispec_dir + "/input/abundances/molecular_symbols.dat"
    #chemical_elements = ispec.read_chemical_elements(chemical_elements_file)
    #isotope_file = ispec_dir + "input/isotopes/SPECTRUM.lst"
    #isotopes = ispec.read_isotope_data(isotope_file)
    #linelist_free_loggf = None
    #free_params = []
    #f = open(ispec_dir+mySamOut_dir+objName+"_res_L2LSynthAbund.txt","w")
    #f.write('Element\twl (nm)\tTurbocode\t[X/H]\te[X/H]\t[X/Fe]\te[X/Fe]\n')
    #for i, line in enumerate(linemasks): #Calculate abundance for each line
    #    if line['element'][-1] == ' ' or int(line['element'][-1]) > 2:
    #        continue
    #    #if elem[:-2] == 'CO':
    #    #    continue
    #    element_name = line['element'][:-2] # Fe 1 -> Fe, Sc 2 -> Sc, etc.
    #    free_abundances = ispec.create_free_abundances_structure([element_name], chemical_elements, solar_abundances)
    #    free_abundances['Abund'] += initial_MH # Scale to metallicity
    #    logging.info("CALCULATING SYNTHETIC [" + element_name + "/Fe] at " + str(np.round(line['wave_peak'],3)) + " nm (line #" + str(i+1) + "/"+str(len(linemasks))+").")
    #    individual_line_regions = linemasks[i:i+1]
    #    segments = ispec.create_segments_around_lines(individual_line_regions, margin=0.25)
    #    wfilter = ispec.create_wavelength_filter(star_spectrum, regions=segments) # Only use the segment
    #    if len(star_spectrum[wfilter]) == 0 or np.any(star_spectrum['flux'][wfilter] == 0):
    #        continue
    #    obs_spec, modeled_synth_spectrum, params, errors, abundances_found, loggf_found, status, stats_linemasks = \
    #            ispec.model_spectrum(star_spectrum[wfilter], star_continuum_model, \
    #            modeled_layers_pack, atomic_linelist, isotopes, solar_abundances, free_abundances, linelist_free_loggf, initial_teff, \
    #            initial_logg, initial_MH, initial_alpha, initial_vmic, initial_vmac, initial_vsini, \
    #            initial_limb_darkening_coeff, initial_R, initial_vrad, free_params, segments=segments, \
    #            linemasks=individual_line_regions, \
    #            enhance_abundances=True, \
    #            use_errors = False, \
    #           vmic_from_empirical_relation = False, \
    #            vmac_from_empirical_relation = False, \
    #            max_iterations=max_iterations, \
    #            tmp_dir = None, \
    #            code=code)
    #    logging.info("SAVING SYNTHETIC [" + element_name + "/Fe] = " + str(np.round(abundances_found[0][5],2)) + " at " + str(np.round(line['wave_peak'],3)) + " nm (line #" + str(i+1) + "/"+str(len(linemasks))+").")
    #    if abundances_found[0][7] > 5.: #Bad uncertainties
    #        abundances_found[0][7] = -1.
    #        abundances_found[0][9] = -1.
    #    f.write('%2s\t%4.3f\t%s\t%1.2f\t%1.2f\t%1.2f\t%1.2f\n' % (abundances_found[0][2], line['wave_peak'], line['turbospectrum_species'], abundances_found[0][3], abundances_found[0][7], abundances_found[0][5], abundances_found[0][9]))
    #    ##--- Save synthetic spectrum -----------------------------------------
    #    synthSpecName = ispec_dir+mySamOut_dir+"L2LSynthSpec/"
    #    if not os.path.exists(synthSpecName):
    #        os.makedirs(synthSpecName)
    #    fOut = open(synthSpecName+objName+"_"+ element_name+"_"+ str(np.round(line['wave_peak'],2))+"nm.txt", "w")
    #    for s in modeled_synth_spectrum:
    #        fOut.write('%9s\t%9s\t%9s\n' % (np.round(s['waveobs'],5),np.round(s['flux'],7), np.round(s['err'],7)))
    #    fOut.close()
    #f.close()
    #print(abundances) 
def SpecSave(star_spectrum):
    f = open(ispec_dir+mySamOut_dir+objName+"_CleanSpec.txt", "w")
    for s in star_spectrum:
        f.write('%9s\t%9s\t%9s\n' % (np.round(s['waveobs'],5),np.round(s['flux'],7), np.round(s['err'],7)))
    f.close()
    
    

if __name__ == '__main__':
    import ispec
    star_spectrum = ispec.read_spectrum(ispec_dir+mySamIn_dir+objName+".txt")
    #star_spectrum = ispec.read_spectrum("/home/max/CallofPhDuty/IntroAndBackUp/DFCyg/DFCyg_norm_00574546.txt")
    #star_spectrum = ispec.read_spectrum("/home/max/CallofPhDuty/iSpec_v20201001/mySample/input/J005252.txt")
    #star_spectrum = ispec.read_spectrum("/home/max/CallofPhDuty/iSpec_v20201001/mySample/input/J005107ob3.txt")
    star_spectrum, star_continuum_model = ContFitAndNorm(star_spectrum)
    #star_spectrum = CosmicFilter(star_spectrum, star_continuum_model)
#    star_spectrum, rv, rv_err = RVCorr(star_spectrum)
#    star_spectrum, snr = SNRErrCalc(star_spectrum)
#    print(SNRErrCalc(star_spectrum))
    SpecSave(star_spectrum)
    #ListCreation()
    linemasks = LineFit(star_spectrum, star_continuum_model, use_ares=False)
    #EWparam(star_spectrum, star_continuum_model, linemasks, code="moog")
    EWabund(star_spectrum, star_continuum_model, linemasks, code="moog")
    pass
