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
initial_teff = 7458.0; initial_logg = 0.5; initial_MH = -0.63
initial_vmic = 3.00; initial_R = 80000.
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
                                use_errors_for_fitting=False)
    #--- Normalize ----------------------------------------------------------
    star_spectrum = ispec.normalize_spectrum(star_spectrum, star_continuum_model, consider_continuum_errors=False)
    star_continuum_model = ispec.fit_continuum(star_spectrum, fixed_value=1.0, model="Fixed value")
    return (star_spectrum, star_continuum_model)
    
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
    
    #--- Radial Velocity correction ------------------------------------------
    logging.info("Radial velocity correction... %.2f +/- %.2f" % (rv, rv_err))
    star_spectrum = ispec.correct_velocity(star_spectrum, rv)
    return (star_spectrum, rv, rv_err)
    
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

def SNRErrCalc(star_spectrum):
    #--- Estimate SNR from flux and errors from SNR --------------------------
    logging.info("Estimating SNR from fluxes (negative flux values will be set to zero)...")
    zeroF = np.where(star_spectrum['flux']<0.0)
    star_spectrum['flux'][zeroF] = 0.0
    num_points = 3
    blue_spec = spec[(spec['waveobs'] >= 662.) & (spec['waveobs'] <= 662.4)]
    green_spec = spec[(spec['waveobs'] >= 663.9) & (spec['waveobs'] <= 664.2)]
    red_spec = spec[(spec['waveobs'] >= 675.8) & (spec['waveobs'] <= 676.4)]
    blue_dump = blue_spec[blue_spec['flux']>0.]
    print('SNR: ', np.nanmedian(blue_dump['flux'])/np.nanstd(blue_dump['flux']), np.nanmedian(green_spec['flux'])/np.nanstd(green_spec['flux']), np.nanmedian(red_spec['flux'])/np.nanstd(red_spec['flux']))
    blue_snr = ispec.estimate_snr(blue_spec['flux'], num_points=num_points)
    green_snr = ispec.estimate_snr(green_spec['flux'], num_points=num_points)
    red_snr = ispec.estimate_snr(red_spec['flux'], num_points=num_points)
    blue_spec = spec[spec['waveobs'] <= 465.]
    green_spec = spec[(spec['waveobs'] >= 465.) & (spec['waveobs'] <= 580.)]
    red_spec = spec[spec['waveobs'] >= 580.]
    star_spectrum['err'] = np.concatenate((blue_spec['flux']/30., green_spec['flux']/90., red_spec['flux']/90.))
    snr = np.array([blue_snr,green_snr,red_snr])
    return (star_spectrum, snr)
    
def ListCreation():
    #--- Calculate theoretical equivalent widths and depths for a linelist --
    Aname, Ateff, Alogg, Amet, Avmic, Acomm = np.loadtxt(ispec_dir + "Spoiler.txt", delimiter='\t', dtype=np.dtype([('name','U8'), ('teff',np.float64), ('logg',np.float64), ('met',np.float64), ('vmic',np.float64), ('comm','U5')]), skiprows=1, unpack=True)
    index = (Aname==objName)
    teff = float(Ateff[index]); logg = float(Alogg[index])
    met = float(Amet[index]); vmic = float(Avmic[index])
    logging.info("CREATING A LINE LIST FOR THE FOLLOWING AP: %4.0f, %1.1f, %1.2f, %1.1f" % (teff, logg, met, vmic))
    alpha = ispec.determine_abundance_enchancements(met)
    model = ispec_dir + "/input/atmospheres/ATLAS9.KuruczODFNEW/"
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
    
    # Line regions
    #line_regions = ispec.read_line_regions(ispec_dir + mySamIn_dir + "Lit_lines_%7s.txt" % objName)

######--- Find linemasks -----------------------------------------------------
    line_regions = ispec.find_linemasks(star_spectrum, star_continuum_model,\
                            atomic_linelist=atomic_linelist, \
                            max_atomic_wave_diff = 0.005, \
                            telluric_linelist=None, \
                            vel_telluric=None, \
                            minimum_depth=0.01, maximum_depth=0.8, \
                            smoothed_spectrum=None, \
                            check_derivatives=False, \
                            discard_gaussian=False, discard_voigt=True, \
                            closest_match=False)
    #f = open(ispec_dir+mySamOut_dir+objName+"_CreatedIntrinBeforefilter.txt", "w")
    #for l in line_regions:
    #    f.write('%.4f\t%.4f\t%.4f\t%s\n' % (l['wave_peak'], l['wave_base'], l['wave_top'], l['element']))
    #f.close()                          
    # Discard bad masks
    flux_peak = star_spectrum['flux'][line_regions['peak']]
    flux_base = star_spectrum['flux'][line_regions['base']]
    flux_top = star_spectrum['flux'][line_regions['top']]
    bad_mask = np.logical_or(line_regions['wave_peak'] <= line_regions['wave_base'], line_regions['wave_peak'] >= line_regions['wave_top'])
    bad_mask = np.logical_or(bad_mask, flux_peak >= flux_base)
    bad_mask = np.logical_or(bad_mask, flux_peak >= flux_top)
    bad_mask = np.logical_or(bad_mask, line_regions['element']=='')
    ##bad_mask = np.logical_or(bad_mask, line_regions['element']=='Fe 1')    
    ##bad_mask = np.logical_or(bad_mask, line_regions['element']=='Fe 2')
    bad_mask = np.logical_or(bad_mask, line_regions['molecule']=='T')
    bad_mask = np.logical_or(bad_mask, line_regions['ion']>2)
    line_regions = line_regions[~bad_mask]
    ID = 'element'
    f = open(ispec_dir+mySamOut_dir+objName+"_CreatedIntrinLineRegion.txt", "w")
    f.write('wave_peak\twave_base\twave_top\tnote\n')
    for l in line_regions:
        f.write('%.4f\t%.4f\t%.4f\t%s\n' % (l['wave_peak'], l['wave_base'], l['wave_top'], l['element']))
    f.close() 
##############################################################################

    ids = [l['note'] for l in line_regions]
    ids_counted = {item:ids.count(item) for item in ids}
    logging.info("LINES READ: " + str(ids_counted))
    logging.info("TOTAL NUMBER: " + str(len(line_regions)))
    
##--- Fit lines ----------------------------------------------------------
    logging.info("Fitting lines...")
    line_regions = ispec.adjust_linemasks(star_spectrum, line_regions, max_margin=0.5)
    linemasks = ispec.fit_lines(line_regions, star_spectrum, star_continuum_model, \
                atomic_linelist = atomic_linelist, \
                max_atomic_wave_diff = 0.005, \
                telluric_linelist = None, \
                smoothed_spectrum = None, \
                check_derivatives = True, \
                vel_telluric = None, discard_gaussian=False, \
                discard_voigt=True, \
                free_mu=True, crossmatch_with_mu=False, closest_match=False)
    
    f = open(ispec_dir+mySamOut_dir+objName+"_LineFit_short.txt", "w")
    for l in linemasks:
        f.write('%5s\t%8s\t%8s\t%8s\n' % (l['element'], np.round(l['wave_nm'],4),np.round(l['ew'],2), np.round(l['ew_err'],2)))
    f.close()
    ispec.write_line_regions(linemasks, ispec_dir+mySamOut_dir+objName+ "_LineFit.txt", extended=True)

    # Discard lines that are not cross matched with the same original element stored in the note
    linemasks = linemasks[linemasks['element'] == line_regions['note']]
    # Select lines that have some minimal contribution in pAGBs
#    atomic_linelist = atomic_linelist[atomic_linelist['theoretical_depth'] >= 0.01]
    # Exclude lines that have not been successfully cross matched with the atomic data
    # because we cannot calculate the chemical abundance (it will crash the corresponding routines)
    linemasks = linemasks[linemasks['wave_nm']!=0.]
    linemasks = linemasks[linemasks['ew']>5.]
    linemasks = linemasks[linemasks['ew']<250.]
    linemasks.sort(order='turbospectrum_species')
    
    ispec.write_line_regions(linemasks, ispec_dir+mySamOut_dir+objName+ "_LineFit.txt", extended=True)
    f = open(ispec_dir+mySamOut_dir+objName+"_LineFit_short.txt", "w")
    for l in linemasks:
        f.write('%5s\t%8s\t%8s\t%8s\n' % (l['element'], np.round(l['wave_nm'],4),np.round(l['ew'],2), np.round(l['ew_err'],2)))
    f.close()
    
    f = open(ispec_dir+mySamOut_dir+objName+"_FitGoodness.txt", "w")
    f.write('lambda (nm)\tArea Ratio (%)\n')
    dev = 0.
    goodLnNum = 0
    for l in linemasks:
        spec = star_spectrum[l['base']:l['top']]
        obs = spec['flux']
        fit = 1.+l['A']*np.exp(-(spec['waveobs']-l['mu'])**2/(2.*l['sig']**2))
#        print('%3.4f, %1.3f' % (l['wave_peak'], np.sum(np.abs(obs-fit))/l['depth']))
        ratio = np.sum(np.abs(obs-fit))*(spec['waveobs'][1]-spec['waveobs'][0])/l['ew']*1.e+4
        if np.abs(ratio) > 0.5:
            l['ew'] = 0.
        else:
            goodLnNum += 1
            dev += np.abs(ratio)
        f.write('%4s\t%3.4f\t%.3f\n' % (l['element'], l['wave_peak'], ratio*100.))
    f.write('OVERALL:\t%.2f (%3i)' % (np.round(dev*100.,2), goodLnNum))
    logging.info("THE OVERALL AREA DEVIATION IS " + str(np.round(dev*100.,2)) + " BASED ON " + str(goodLnNum) + " LINES")
    f.close()
    linemasks = linemasks[linemasks['ew']>0.]
    
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
        spec = star_spectrum[linemasks['peak'][i]-100:linemasks['peak'][i]+100]
        gauss = 1.+linemasks['A'][i]*np.exp(-(spec['waveobs']-linemasks['mu'][i])**2/(2.*linemasks['sig'][i]**2))
        step = spec['waveobs'][100]-spec['waveobs'][99]
#        from_x = 100 - int(6.*linemasks['sig'][i]/step)
#        to_x = 100 + int(6.*linemasks['sig'][i]/step)
        wave_filter_custom = (spec['waveobs'] >= linemasks['wave_base'][i]) & (spec['waveobs'] <= linemasks['wave_top'][i])
        line = spec[wave_filter_custom]
        lineEndsX = [line['waveobs'][0], line['waveobs'][-1]]
##        lineEndsX = [linemasks['wave_base'][i]-linemasks['wave_peak'][i]+100, linemasks['wave_top'][i]-linemasks['wave_peak'][i]+100]
        lineEndsY = [line['flux'][0], line['flux'][-1]]
#        lineEndsY = [1., 1.]
        cont_a = (line['flux'][-1]-line['flux'][0])/(line['waveobs'][-1]-line['waveobs'][0])
        cont_b = (line['waveobs'][-1]*line['flux'][0]-line['waveobs'][0]*line['flux'][-1])/(line['waveobs'][-1]-line['waveobs'][0])
#        cont_a = 0.; cont_b = 1.
        continuum_custom = cont_a*line['waveobs'] + cont_b
        
        fig = plt.figure(figsize=(11,8))
        ax = fig.add_subplot(1, 1, 1)
        plt.xlim([linemasks['wave_peak'][i]-0.25,linemasks['wave_peak'][i]+0.25])
        plt.ylim([0.4,1.2])
        plt.xlabel('$\lambda$ (nm)')
        plt.title(linemasks['element'][i] + ' (%i/%i), EW = %3.1f mA' % (i+1,len(linemasks),linemasks['ew'][i]))
        ax.plot(spec['waveobs'], spec['flux'], '-k')
        ax.plot(lineEndsX, lineEndsY, 'co', ms=5)
        ax.plot(lineEndsX, lineEndsY, 'c-', lw=3)
#        ax.plot(spec['waveobs'][from_x:to_x], gauss[from_x:to_x], '-r')
#        ax.axvline(linemasks['wave_base'][i], c='yellow', zorder=0)
#        ax.axvline(linemasks['wave_top'][i], c='yellow', zorder=0)
        ax.axvline(linemasks['wave_peak'][i], c='red', zorder=1)
        ax.axhline(1., c='gray', ls='--', zorder=0)
        ax.fill_between(line['waveobs'], line['flux'], continuum_custom, color='cyan', zorder=0)
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
    model = ispec_dir + "input/atmospheres/ATLAS9.Castelli/"

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

    results = ispec.model_spectrum_from_ew(linemasks, modeled_layers_pack, \
                        solar_abundances, initial_teff, initial_logg, initial_MH, initial_alpha, initial_vmic, \
                        free_params=["teff", "logg", "vmic"], \
                        adjust_model_metalicity=True, \
                        max_iterations=max_iterations, \
                        enhance_abundances=True, \
                        #outliers_detection = "robust", \
                        #outliers_weight_limit = 0.90, \
                        outliers_detection = "sigma_clipping", \
                        sigma_level = 3, \
                        tmp_dir = None, \
                        code=code)
    params, errors, status, x_over_h, selected_x_over_h, fitted_lines_params, used_linemasks = results
    
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
    idx = np.isfinite(x_over_h)
    Fe_line_regions = Fe_line_regions[idx]; x_over_h = x_over_h[idx]
    ind1 = np.where(Fe_line_regions['element']=='Fe 1')
    ind2 = np.where(Fe_line_regions['element']=='Fe 2')
    
    ## v_mic correction
    plt.xlim([-6.,-4.2])
    plt.ylim([np.nanmean(x_over_h)-1,np.nanmean(x_over_h)+1])
    plt.xlabel("reduced EW")
    plt.ylabel("[M/H]")
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
    plt.ylabel("[M/H]")
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
    for elem in linemasks['element']:
        if (elem not in abundTargets and elem != ''):
            abundTargets.append(elem)
    logging.info("HERE ARE OUR CONTESTANTS:" + str(abundTargets))
    
    model = ispec_dir + "input/atmospheres/ATLAS9.Castelli/"
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

    ##--- EW abund -----------------------------------------------------------
    f = open(ispec_dir+mySamOut_dir+objName+"_res_Abund.txt", "w")
    f.write('Element\tN\t[X/H]\te[X/H]\t[X/Fe]\te[X/Fe]\n')
    for elem in abundTargets: #Calculate abundance for each element
        element_name = elem[:-2] # Fe 1 -> Fe, Sc 2 -> Sc, etc.
        logging.info("CALCULATING [" + element_name + "/Fe].")
        elem_line_regions = [line for line in linemasks if line[0]==elem]
        elem_line_regions = np.array(elem_line_regions)
        atmosphere_layers = ispec.interpolate_atmosphere_layers(modeled_layers_pack, {'teff':initial_teff, 'logg':initial_logg, 'MH':initial_MH, 'alpha':initial_alpha}, code=code)
        spec_abund, normal_abund, x_over_h, x_over_fe = ispec.determine_abundances(atmosphere_layers, \
            initial_teff, initial_logg, initial_MH, initial_alpha, elem_line_regions, solar_abundances, microturbulence_vel = initial_vmic, \
            verbose=1, code=code)
        ##--- Save results --------------------------------------------------
        logging.info("Saving results...")
        f.write('%2s\t%i\t%1.2f\t%1.2f\t%1.2f\t%1.2f\n' % (elem, len(elem_line_regions), np.mean(x_over_h), 0.00, np.mean(x_over_fe), 0.00))
    f.close()
    
    ##--- SS L2L abund --------------------------------------------------------
    initial_vmac = ispec.estimate_vmac(initial_teff, initial_logg, initial_MH)
    initial_vsini = 1.6; initial_limb_darkening_coeff = 0.6; initial_vrad = 0
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
        if abundances_found[0][7] > 5.:
            abundances_found[0][7] = -1.
            abundances_found[0][9] = -1.
        f.write('%2s\t%4.3f\t%s\t%1.2f\t%1.2f\t%1.2f\t%1.2f\n' % (abundances_found[0][2], line['wave_peak'], line['turbospectrum_species'], abundances_found[0][3], abundances_found[0][7], abundances_found[0][5], abundances_found[0][9]))
        ##--- Save synthetic spectrum -----------------------------------------
        synthSpecName = ispec_dir+mySamOut_dir+"L2LSynthSpec/"
        if not os.path.exists(synthSpecName):
            os.makedirs(synthSpecName)
        fOut = open(synthSpecName+objName+"_"+ element_name+"_"+ str(np.round(line['wave_peak'],2))+"nm.txt", "w")
        for s in modeled_synth_spectrum:
            fOut.write('%9s\t%9s\t%9s\n' % (np.round(s['waveobs'],5),np.round(s['flux'],7), np.round(s['err'],7)))
        fOut.close()
    f.close()

def SpecSave(star_spectrum):
    f = open(ispec_dir+mySamOut_dir+objName+"_CleanSpec.txt", "w")
    for s in star_spectrum:
        f.write('%9s\t%9s\t%9s\n' % (np.round(s['waveobs'],5),np.round(s['flux'],7), np.round(s['err'],7)))
    f.close()
    
    

if __name__ == '__main__':
    LinUpd()
    import ispec
    star_spectrum = ispec.read_spectrum(ispec_dir+mySamIn_dir+objName+".txt")
    star_spectrum, star_continuum_model = ContFitAndNorm(star_spectrum)
    star_spectrum = CosmicFilter(star_spectrum, star_continuum_model)
#    star_spectrum, rv, rv_err = RVCorr(star_spectrum)
#    star_spectrum, snr = SNRErrCalc(star_spectrum)
    SpecSave(star_spectrum)
#    ListCreation()
    linemasks = LineFit(star_spectrum, star_continuum_model, use_ares=False)
#    EWparam(star_spectrum, star_continuum_model, linemasks, code="moog")
#   EWabund(star_spectrum, star_continuum_model, linemasks, code="moog")
    pass
