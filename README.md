# E-Ispec_Optical
Semi-automated python wrapper for high-resolution optical spectroscopic analysis of evolved stars

###########################################################################

```
TO RUN A NEW STAR YOU NEED TO:

a) add the source's spectrum ("waveobs\tflux\terr\n") and the line regions ("wave_peak\twave_base\twave_top\tnote\n") to '.../mySample/input/' folder,
b) open the terminal in the root folder and type 'python Flug_opt.py 7-symbol-star-ID' (for example, "J005107").
```
```
a) images/                         = iSpec logos (PNG and GIF),
b) input/                          = iSpec input files: solar abundances, model atmospheres, model atmosphere grid and mini-grid (ATLAS, MARCS), line lists, and template spectra
c) isochrones/                     = Yonsei-Yale isochrone plotter,
d) ispec/                          = all PySpec essentials,
e) synthesizer/                    = radiative transfer codes,
f) .git/                           = iSpec git-related folder,
g) mySample/                       = input and output files for E-iSpec,
h) dev-requirements.txt            = Python requirements,
i) example.py                      = different examples of working with the PySpec functions,
j) Flug_linelist.py                = a script to create atomic linelist,
k) Flug_opt.py                     = the main E-iSpec script,
l) interactive.py                  = GUISpec essential,
m) ispec.log                       = a concatenation of all the terminal logs (only iSpec-specific logs),
n) iSpec.command                   = a bash script invoking GUISpec,
o) LICENSE                         = iSpec license,
p) Makefile                        = subroutines initial builder (check iSpec installation manual),
q) pytest.ini                      = Python path testing,
r) README.md                       = iSpec readme,
```
