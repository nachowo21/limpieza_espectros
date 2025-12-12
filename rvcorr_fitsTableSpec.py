#!/usr/bin/python3
# -*-encoding:utf-8 -*-

r'''
	RVCORR
	  Computes and correct doppler shift of a given spectrum using a
	  grid of synthetic spectra
	  
	USAGE
	  Details concerning the calling options are given below. --rin and --rout allow to
	  provide a file defining regions to be used to perform the cross correlation. If 
	  none of these areguments is provided, the script will use whe whole observed spectra
	  to calculate the CCF. If --rin file is provided, an invividual analysis is performed
	  in each region and the resulting RV is the mean of all of them. In addition this allow
	  to obtain an error estimation from the std of these measurements. If --rout file is
	  provided, the CCF is calculated using the whole observed spectra but the regions
	  indicated in the file. The use of these card is mutually exclusive. Note that while
	  defining these inclusion/exclusion they must be compared with the observed spectrum
	  as it comes.
	  
	CHANGES
	  -Add cubic spline minimization to find the maximum of the CCF. AR Jun 1 2017.
	  -Now template fundamental parameters can be read from the fits files if present
	   in the header. AR Jun 5 2017
	  -Add code to permit the calculations to be done in several user defined
	   wavelenght intervals. AR Jun 5 2017
	  -Add code to manage when the observed spectra does not have CD1_1 card. In this case
	   the script does not try to edit/add it into the RV corrected spectra to be written.
	   AR Jun 8 2017.
	  -Now the curves and shaded regions have number labels to be identifyed. AR Jun 8 2017
	  -Regions can be masked (zeroed) from the observed template via the --rout option.
	   AR Jun 12 2017.
	   - Modified to read spectra in fits table format (wave, flux, sigma_flux) and write
	    corrected spectrum in the same format
	    AR may 2025
'''

import matplotlib as mpl
mpl.rcParams['font.family']="serif"
mpl.rcParams['axes.linewidth'] = 0.85
from matplotlib.ticker import AutoMinorLocator
import astropy.io.fits as pf
from astropy.wcs import WCS
import numpy as np
import matplotlib.pylab as plt
import sys
import matplotlib.gridspec as gridspec
from scipy.interpolate import InterpolatedUnivariateSpline,interp1d
import argparse
import os
import scipy.optimize as opt


# Read command line inputs
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("InFile",help="Input spectrum to be corrected")
parser.add_argument("Grid",help="Name of the folder containin the grid of comparisson spectra")
parser.add_argument("-n","--noplot",help="Graphic output: no plot will be displayed",action="store_true")
parser.add_argument("--vmin",help="Minimum velocity to search for",type=float)
parser.add_argument("--vmax",help="Maximum velocity to search for",type=float)
parser.add_argument("--rin",help="File with regions to include",type=str)
parser.add_argument("--rout",help="File with regions to exclude",type=str)
args = parser.parse_args()

infile=args.InFile

grid=args.Grid


#===============EDIT COLUMN NAMES and table format==============
# Set here the label that the different colums
# have in the fits tables with the spectra
wave_col_name = "LAMBDA"
flux_col_name = "CFLUX"
err_flux_col_name = "flux_norm_err"



#  Abrir espectro observado: fits table format. Column names: lambda, flux, sigma_flux
hdulist_obs = pf.open(infile)
tt = hdulist_obs[1].data
# tt = hdulist_obs[0].data
# hdr = hdulist_obs[0].header
# n_pix = tt.shape[0]
# w = WCS(hdr, naxis = 1)
# wave = w.pixel_to_world(np.arange(n_pix)).value
wave = tt[wave_col_name]*1e4 #-3  # este es para molecfit !!!!!!!!!!LE RESTO EL 3 PORQUE AL PARECER ESTA CONSTANTE LO ARREGLA
flux = tt[flux_col_name] # este tambien es para molecfit
#flux = tt
sigma_flux = tt[flux_col_name]

#==============================================





def oneGauss(x,A,mu,sig):
	zg = (x-mu)**2/sig**2
	gg = A*np.exp(-zg/2.0)
	return gg







#plt.plot(wave,flux)
#plt.show()


#  Leer lista de archivos fits de grilla de espectros sinteticos
files_list = os.listdir(grid)
templates_list=[]
for elem in files_list:
	if ".fits" in elem:
		templates_list.append(elem)
if len(templates_list)==0:
	print("\n[Error] No templates available in folder "+grid+"\n")
	sys.exit()
else:
	print("[Info] Number of templates: ",len(templates_list))

# Store in lists the fluxes and parameters
# of files in the grid
fluxes_sint = [None]*len(templates_list)
params_list = [None]*len(templates_list)
for i in range(len(templates_list)):
	if i==0:
		flux_sint, header_sint = pf.getdata(grid+"/"+templates_list[i], header=True)
		wcs = WCS(header_sint)
		index = np.arange(header_sint['NAXIS1'])
		wavelength = wcs.wcs_pix2world(index[:,np.newaxis], 0)
		wave_sint = wavelength.flatten()
		fluxes_sint[i]=flux_sint
		
	else:
		flux_sint,header_sint = pf.getdata(grid+"/"+templates_list[i], header=True)
		fluxes_sint[i]=flux_sint
	
	# Ver si hay parametros fisicos anotados en el header
	hkeys = header_sint.keys()
	hparams=[np.nan]*4
	if "TEFF" in hkeys:
		hparams[0]=float(header_sint["TEFF"])
	if "LOGG" in hkeys:
		hparams[1]=float(header_sint["LOGG"])
	if "MH" in hkeys:
		hparams[2]=float(header_sint["MH"])
	if "ALFE" in hkeys:
		hparams[3]=float(header_sint["ALFE"])
	params_list[i]=hparams


# Calculate the minimum and maximum velocity possible to calculate with the provided grid
vlight = 2.997925e5
dif_range = wave_sint[-1]-wave_sint[0]-(wave[-1]-wave[0])
if dif_range<=0.0:
	print("[Error] Wavelength coverage of template is too short")
	print("\tTemplate/range: ",templates_list[i],"/", wave_sint[-1]-wave_sint[0])
	print("\tObs spectrum: ",infile)
	print(wave[-1]-wave[0])
	sys.exit()
v_min =  vlight*(wave[-1]/wave_sint[-1]-1.0)+4.0
v_max =  vlight*(wave[0]/wave_sint[0]-1.0)-4.0
if (args.vmin!=None):
	v_min=args.vmin
if (args.vmax!=None):
	v_max=args.vmax
if (args.vmin==None) & (args.vmax==None):
	print("[Info] Provided grid allows to compute velocity shifts in(%8.2f,%8.2f)"%(v_min,v_max))
if (args.vmin!=None) & (args.vmax!=None):
	print("[Info] Inputs allow to compute velocity shifts in(%8.2f,%8.2f)"%(v_min,v_max))


#  Definicion de la funcion de croscorrelacion
def cross_correlate(wobs,fobs,wsin,fsin,vmin=v_min,vmax=v_max,deltaV=0.5):
	vels = np.arange(vmin, vmax, deltaV)
	ccf = np.zeros(len(vels))
	for i, vel in enumerate(vels):
		factor = np.sqrt( (1.0+vel/vlight) / (1.0-vel/vlight) )
        #f_sin_shift = interp1d(wsin*factor, fsin, bounds_error=True)
		f_sin_shift = interp1d(wsin*factor, fsin, bounds_error=False, fill_value=0.0)
		ccf[i] = np.sum(fobs*f_sin_shift(wobs))
	return vels,ccf

#  Attempt level-off normalization of input spectrum  --> by-hand recipe, look for smth better!!!
#median_flux_obs=np.percentile(flux[flux>0.7],75)
median_flux_obs = np.nanmedian(flux[(flux>0) & np.isfinite(flux)])
if median_flux_obs==0: median_flux_obs=1.0
# Define regions to compute cross correlation
if (args.rin!=None):
	regions = np.genfromtxt(args.rin)
	if np.shape(regions)==(2,):
		regions=[regions]
else:
	regions = np.array([[np.min(wave),np.max(wave)]])

# Define regions to exclude from the analysis
imask_in=np.isfinite(wave)
if (args.rout!=None):
	regions_out = np.genfromtxt(args.rout)
	if np.shape(regions_out)==(2,):
		regions_out=[regions_out]
	for region in regions_out:
		i_mask =  (wave>=region[0]) & (wave<=region[1]) 
		imask_in = imask_in & ~i_mask
flux_masked = np.where(imask_in,flux,0.0)

#  Make a preliminary cross correlation against a single template (choosen randomly)
#  to roughly set the observed spectrum into rest frame
i_prelim_tpl = np.random.randint(len(templates_list))
vels=[]
vvs=[]
ccfs=[]
for region in regions:
	ireg = (wave>=region[0]) & (wave<=region[1]) 
	vv,ccf = cross_correlate(wave[ireg],flux_masked[ireg]/median_flux_obs,wave_sint,fluxes_sint[i_prelim_tpl])
	vcorr_prelim = vv[np.argmax(ccf)]
	vels.append(vcorr_prelim)
	vvs.append(vv)
	ccfs.append(ccf)
vcorr_prelim = np.median(vels)
print("[Info] Preliminary Vobs:%8.2f"%vcorr_prelim)



##
#   Make a diagnostic plot of the whole process
#
fig = plt.figure(1,figsize=(12,7),facecolor="white")
fig.subplots_adjust(left=0.065,bottom=0.07,right=0.98,top=0.98,hspace=0.0,wspace=0.0)
gs1 = gridspec.GridSpec(76, 40)
#fig.canvas.set_window_title('Radial velocity correction ['+infile+"]")
mpl.rcParams.update({'font.size': 10})

# Plot first Vcorr
ax1=fig.add_subplot(gs1[0:30,0:18])
ax1.set_xlim(v_min,v_max)
ax1.set_xlabel("RV (km/s)")
ax1.set_ylabel("Cross Corr Norm.")
for i in range(len(vvs)):
	ax1.plot(vvs[i], ccfs[i]/np.max(ccfs[i]),color="olivedrab")
	v_label = v_min+0.07*(v_max-v_min) 
	i_label=np.argmin(np.absolute(v_label-vvs[i]))
	ccfy_label = ccfs[i][i_label]/np.max(ccfs[i])
	ax1.text(v_label,ccfy_label,str(i),color="black",fontsize=10)
	ax1.axvline(x=vels[i],ls="--",lw=0.6,color="black")
ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
xl=ax1.get_xlim()
yl=ax1.get_ylim()
ax1.text(xl[0]+0.7*(xl[1]-xl[0]),yl[0]+0.85*(yl[1]-yl[0]),"Cross-correlation\nguess template",fontsize=11)


# Calculate chisquare of obseved rv corrected spectra against grid
#  We consider only user defineded regions, discarding those indicated 
fcor = np.sqrt( (1.0-vcorr_prelim/vlight) / (1.0+vcorr_prelim/vlight) )
wave_corr = wave*fcor
w1 = np.max((np.min(wave_corr),np.min(wave_sint)))
w2 = np.min((np.max(wave_corr),np.max(wave_sint)))
i_cm = (wave_sint>=w1) & (wave_sint<=w2)
spl = InterpolatedUnivariateSpline(wave_corr, flux,k=3) # interpolate obs spec in common range with template
flux_obs_corr_int = spl(wave_sint[i_cm])/median_flux_obs

# Define regions to exclude from the analysis considering the fcor 
imask_in_corr=np.isfinite(wave_sint[i_cm])
if (args.rout!=None):
	regions_out = np.genfromtxt(args.rout)
	if np.shape(regions_out)==(2,):
		regions_out=[regions_out]
	for region in regions_out:
		i_mask =  (wave_sint[i_cm]>=region[0]*fcor) & (wave_sint[i_cm]<=region[1]*fcor) 
		imask_in_corr = imask_in_corr & ~i_mask

i_chi = np.zeros(len(wave_sint[i_cm]))
for region in regions:
	icond = (wave_sint[i_cm]>=region[0]*fcor) & (wave_sint[i_cm]<=region[1]*fcor)
	i_chi = i_chi + icond
i_chi = np.array(i_chi,dtype=bool) & imask_in_corr

chi2=[None]*len(templates_list)
for k in range(len(fluxes_sint)):
	chi2[k] = np.sum( (flux_obs_corr_int[i_chi]-fluxes_sint[k][i_cm][i_chi])**2) / (np.sum(flux_obs_corr_int[i_chi])-3.0)
chi2=np.array(chi2)
i_best_hit = np.argmin(chi2)
print("[Info] Best template chisquare: ", chi2[i_best_hit])



#  Make correlation with the best template
vels=[]
vvs=[]
ccfs=[]
for region in regions:
	ireg = (wave>=region[0]) & (wave<=region[1])
	vv,ccf = cross_correlate(wave[ireg],flux_masked[ireg]/median_flux_obs,wave_sint,fluxes_sint[i_best_hit],vmin=vcorr_prelim-35,vmax=vcorr_prelim+35,deltaV=0.025)
	vcorr_best_discrete = vv[np.argmax(ccf)]
	vels.append(vcorr_best_discrete)
	vvs.append(vv)
	ccfs.append(ccf)
vcorr_best_discrete = np.median(vels)
vvs=np.array(vvs)
ccfs=np.array(ccfs)

##
#   Luego de probar algunos metodos para encontrar el maximo con mas exactitud, concluyo que lo mejor es
#   minimizar una funcion spline cubica construida a partir del 20% mayor de la ccf. El fit de Gaussiana 
#   no resulta tan bien porque incluso la parte superior de la ccf puede ser levemente asimetrica
#
vels_best=np.empty(len(vels))
for i in range(len(vels)):
	ccf_norm = ccfs[i]/np.max(ccfs[i])
	ii = ccf_norm > np.percentile(ccf_norm,85)
	spl_ccf = InterpolatedUnivariateSpline(vvs[i][ii],ccf_norm[ii],k=3)
	spl_curve = spl_ccf(np.linspace(np.min(vvs[i][ii]),np.max(vvs[i][ii]),1000))
	fm = lambda x: -spl_ccf(x)
	vcorr_minimized = opt.minimize_scalar(fm, bounds=(np.min(vvs[i][ii]),np.max(vvs[i][ii])),method="bounded")
	vcorr_best=vcorr_minimized.x
	vels_best[i]=vcorr_best
vcorr_best = np.median(vels_best)
err_vcorr_best = np.std(vels_best)
print("[Info] ccf Vobs:%9.3f"%vcorr_best_discrete)
print("[Info] Final Vobs:%9.3f"%vcorr_best)
if len(regions)>1:
	print("[Info] Vobs Err:%9.3f"%err_vcorr_best)

fcor = np.sqrt( (1.0-vcorr_best/vlight) / (1.0+vcorr_best/vlight) )
wave_best = wave*fcor


# Compute template autocorrelation
vv0,ccf0 = cross_correlate(wave_sint[i_cm],fluxes_sint[i_best_hit][i_cm],wave_sint,fluxes_sint[i_best_hit],vmin=-35,vmax=35, deltaV=0.1)

# Plot best Vcorr with all correlation functions
ax2=fig.add_subplot(gs1[0:30,22:40])
ax2.set_xlabel("RV (km/s)")
ax2.set_ylabel("Cross Corr Norm.")
for i in range(len(ccfs)):
	ax2.plot(vvs[i], ccfs[i]/np.max(ccfs[i]),color="olivedrab",label="Cross-correlation")
handles, labels = ax2.get_legend_handles_labels()
handles_gral=[]
labels_gral=[]
handles_gral.append(handles[0])
labels_gral.append(labels[0])
ax2.plot(vv0+vcorr_best, ccf0/np.max(ccf0),color="black",ls="--",label="Auto-correlation\ntemplate")
handles, labels = ax2.get_legend_handles_labels()
handles_gral.append(handles[len(ccfs)])
labels_gral.append(labels[len(ccfs)])
ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
ccf_mins=[]
for i in range(len(ccfs)):
	ccf_mins.append(np.min((np.min(ccfs[i]/np.max(ccfs[i])),np.min(ccf0/np.max(ccf0)))))
yplotmin=np.min(ccf_mins)
yl_i = yplotmin-0.05*(1.0-yplotmin)
yl_s = 1.0+0.1*(1.0-yplotmin)
ax2.set_ylim(yl_i,yl_s)
xl=ax2.get_xlim()
yl=ax2.get_ylim()
ax2.legend(handles_gral,labels_gral,loc=2,frameon=False,fontsize=11)
#ax2.text(xl[0]+0.68*(xl[1]-xl[0]),yl[0]+0.85*(yl[1]-yl[0]),"Cross-correlation\nbest template")
ax2.text(xl[0]+0.60*(xl[1]-xl[0]),yl[0]+0.7*(yl[1]-yl[0]),"Vobs =%9.3f km/s"%vcorr_best,color="tomato",fontsize=11)
if len(regions)>1:
	ax2.text(xl[0]+0.60*(xl[1]-xl[0]),yl[0]+0.63*(yl[1]-yl[0]),"Err Vobs =%9.3f km/s"%err_vcorr_best,color="tomato",fontsize=11)



# Plot corrected spectrum vs template
ax3=fig.add_subplot(gs1[36:75,0:40])
ax3.set_xlabel("lambda (Angstrom)")
ax3.set_ylabel("Normalized flux")
#ax3.set_xlim(np.min(wave_sint),np.max(wave_sint))
ax3.set_xlim(np.min(wave_best),np.max(wave_best))
minx = np.min([np.min(flux/median_flux_obs)+0.04,0.7])
ax3.set_ylim(minx,1.15)
ax3.plot(wave_best,flux/median_flux_obs,lw=0.5,color="black",label="Observed")
ax3.plot(wave_sint,fluxes_sint[i_best_hit],lw=0.5,color="red",label="Template")
#ax3.plot(wave_best,flux-(flux_shift-1.0),lw=0.5,color="blue",label="Residual")
ax3.xaxis.set_minor_locator(AutoMinorLocator(5))
name_best=templates_list[i_best_hit]
xl=ax3.get_xlim()
yl=ax3.get_ylim()
ax3.text(xl[0]+0.5*(xl[1]-xl[0]),yl[0]+0.15*(yl[1]-yl[0]),"Best template: %35s"%name_best,fontsize=11,color="maroon")
ax3.text(xl[0]+0.5*(xl[1]-xl[0]),yl[0]+0.07*(yl[1]-yl[0]),"Teff=%6.1f"%params_list[i_best_hit][0],fontsize=11)
ax3.text(xl[0]+0.6*(xl[1]-xl[0]),yl[0]+0.07*(yl[1]-yl[0]),"log(g)=%5.2f"%params_list[i_best_hit][1],fontsize=11)
ax3.text(xl[0]+0.7*(xl[1]-xl[0]),yl[0]+0.07*(yl[1]-yl[0]),"[M/H]=%5.2f"%params_list[i_best_hit][2],fontsize=11)
ax3.text(xl[0]+0.8*(xl[1]-xl[0]),yl[0]+0.07*(yl[1]-yl[0]),"[a/Fe]=%5.2f"%params_list[i_best_hit][3],fontsize=11)

# Highlight used regions except if whole spectrum is used
if len(regions)>1:
	for i,region in enumerate(regions):
		ax3.fill_between(np.linspace(region[0],region[1],1000),yl[0],yl[1],lw=0,color="cornflowerblue",alpha=0.4)
		ax3.text(region[0],yl[0]+0.90*(yl[1]-yl[0]),str(i),color="black",fontsize=10)

# Highlight excluded regions
if (args.rout!=None):
	regions_out = np.genfromtxt(args.rout)
	if np.shape(regions_out)==(2,):
		regions_out=[regions_out]
	for region in regions_out:
		ax3.fill_between(np.linspace(region[0]*fcor,region[1]*fcor,500),yl[0],yl[1],lw=0,color="tomato",alpha=0.4)
ax3.set_xlim(xl)
ax3.set_ylim(yl)

# Highlight included regions
if len(regions) > 0:
	for region in regions:
		ax3.axvspan(region[0], region[1], color='cornflowerblue', alpha=0.1)


# Output or interactive display
if args.noplot:
	fig.savefig(infile.split(".fits")[0]+"_vcorr.png")
	plt.close(fig)
else:
	plt.show()


hdulist_obs.close()


outfile = infile.split(".fits")[0]+"_vcorr.fits"
col1 = pf.Column(name='lambda', format='D', array=wave_best)
col2 = pf.Column(name='flux', format='D', array=flux)
#col3 = pf.Column(name='sigma_flux', format='D', array=sigma_flux)
hdu = pf.BinTableHDU.from_columns([col1,col2])#,col3
hdu.header['DopCor']=(vcorr_best,'Vrad correction applied')
hdu.writeto(outfile,overwrite=True)
print("[Info] Corrected spectrum written in: ",outfile)

salida = open(infile.split(".fits")[0]+"_RV.dat","w")
if len(regions)==1:
	linsal = "%25s  %9.3f %9.3f  %6.1f  %5.2f  %5.2f  %5.2f \n" % (infile.split(".fits")[0],vcorr_best,np.nan,params_list[i_best_hit][0],params_list[i_best_hit][1],params_list[i_best_hit][2],params_list[i_best_hit][3])
	salida.write(linsal)
else:
	linsal = "%25s  %9.3f %9.3f  %6.1f  %5.2f  %5.2f  %5.2f \n" % (infile.split(".fits")[0],vcorr_best,err_vcorr_best,params_list[i_best_hit][0],params_list[i_best_hit][1],params_list[i_best_hit][2],params_list[i_best_hit][3])
	salida.write(linsal)
salida.close()
print("[Info] Results written in ",infile.split(".fits")[0]+"_RV.dat\n")






