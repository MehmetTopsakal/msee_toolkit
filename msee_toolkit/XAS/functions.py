

import os,sys,datetime
import glob,linecache,shutil
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt

from larch.io import read_ascii,read_athena
from larch.xafs import find_e0,pre_edge,autobk,xftf
from larch import Group
import larch

from copy import deepcopy




def get_fl(pattern,mode=['ISS']):
    
    """
    This function searches files in a directory and sorts by experiment start time
    """
    
    fl_in = []
    fl = sorted(glob.glob(pattern))
    
    for e,f in enumerate(fl):

        try:
        
            if mode[0] == 'ISS':
                l = linecache.getline(f, mode[1])
                dt = datetime.datetime.strptime('%s_%s'%(l.split()[2],l.split()[3]),
                                                "%m/%d/%Y_%H:%M:%S")
            if mode[0] == 'ISS_old':
                l = linecache.getline(f, mode[1])
                dt = datetime.datetime.strptime('%s_%s'%(l.split()[3],l.split()[4]),
                                                "%m/%d/%Y_%H:%M:%S")
            elif mode[0] == 'ISS_2021_3':
                l = linecache.getline(f, mode[1])
                dt = datetime.datetime.strptime('%s_%s'%(l.split()[2],l.split()[3]),
                                                "%m/%d/%Y_%H:%M:%S")
            elif mode[0] == 'QAS':
                l = linecache.getline(f, mode[1])
                dt = datetime.datetime.strptime('%s_%s'%(l.split()[3],l.split()[4]),
                                                "%m/%d/%Y_%H:%M:%S")
            elif mode[0] == 'BMM':
                l = linecache.getline(f, mode[1])
                dt = datetime.datetime.strptime(l,"# Scan.start_time: %Y-%m-%dT%H:%M:%S\n") 
            elif mode[0] == '12BM':
                l = linecache.getline(f, mode[1])  
                dt = datetime.datetime.strptime(l,"#D %a %b %d %H:%M:%S %Y \n")            
            elif mode[0] == '20ID':
                l = linecache.getline(f, mode[1]).split()
                dt = datetime.datetime.strptime('%s_%s_%s'%(l[9],l[10],l[11][0:2]),
                                                "%m/%d/%Y_%I:%M:%S_%p") 
                
            fl_in.append([dt.timestamp(),dt,f])   

        except Exception as exc:
            print(exc)
            print('Unable to read %s'%(f))

        
    # sort by timestamp
    fl_in.sort(key=lambda x: x[0])
    fl_out = [[ i[1].isoformat() ,i[2]] for e,i in enumerate(fl_in)]
    
    return fl_out



def read_as_ds(fl_in,mode='ISS',Eshift=0,
               imin=0,imax=-1,plot=True,legend=False,plot_ref=True,xlim=None,
               cut=0):
    Es = []
    MUs_f = []
    MUs_r = []
    
    d0 = np.loadtxt(fl_in[0][1],unpack=True)
    
    read_data = []
    for i in fl_in:
        d = np.loadtxt(i[1],unpack=True)
        if mode == 'ISS':
            MUs_f.append(d[4]/d[1])
            MUs_r.append(-np.log(d[3]/d[2]))    
            Es.append(d[0])
        if mode == 'QAS':
            MUs_f.append(d[1]/d[2])
            MUs_r.append(np.log(d[1]/d[3]))    
            Es.append(d[0])
        elif mode == 'BMM':        
            MUs_f.append(d[3])
            MUs_r.append(-np.log(d[6]/d[4]))    
            Es.append(d[0])
        elif mode == '12BM':        
            MUs_f.append(d[9]/d[2])   
            MUs_r.append(d[7]/d[2])    
            Es.append(d[0])   
        elif mode == '20ID_98':  
            MUs_f.append(d[9]/d[8])
            #MUs_r.append(d[9]/d[8]) #for compatibility issues
            Es.append(d[0])         
        elif mode == '20ID_186':  
            MUs_f.append(d[18]/d[6])
            #MUs_r.append(d[18]/d[6]) #for compatibility issues
            Es.append(d[0])    
        elif mode == '20ID_128':  
            MUs_f.append(d[12]/d[8])
            #MUs_r.append(d[12]/d[8]) #for compatibility issues
            Es.append(d[0])    
        elif mode == '20ID_108':  
            MUs_f.append(d[10]/d[8])
            #MUs_r.append(d[10]/d[8]) #for compatibility issues
            Es.append(d[0])
        elif mode == '20ID_65ref':  
            MUs_f.append(-np.log(d[6]/d[5]))
            MUs_r.append(-np.log(d[6]/d[5]))    
            Es.append(d[0]) 

    if plot:

        if plot_ref and MUs_r != []:
            fig = plt.figure(figsize=(12,6),dpi=96)
            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2)
        else:
            fig = plt.figure(figsize=(8,6),dpi=96)
            ax1 = fig.add_subplot(1,1,1)
        

        for e,i in enumerate(MUs_f):
            ax1.plot(Es[e],i,label=fl_in[e][1]+' (ind:%d time:%s)'%(e,fl_in[e][0]))
        ax1.set_xlabel('E (eV)')
        ax1.set_ylabel('$\mu(E)$')
        ax1.set_title('Fluoresence')
        ax1.axvline(x=Es[e][imin],linestyle='--',color='k')
        ax1.axvline(x=Es[e][imax],linestyle='--',color='k')

        if legend:
            ax1.legend(fontsize=8,loc='best',frameon=False)

        if plot_ref and MUs_r != []:
        
            for e,i in enumerate(MUs_r):
                ax2.plot(Es[e],i,label=fl_in[e][1]+' (ind:%d time:%s)'%(e,fl_in[e][0]))
            ax2.set_xlabel('E (eV)')
            ax2.set_title('Reference')   
            ax2.axvline(x=Es[e][imin],linestyle='--',color='k')
            ax2.axvline(x=Es[e][imax],linestyle='--',color='k')
            ax2.set_xlim(xlim)
            
        ax1.set_xlim(xlim)
        
    
    # for spectra that have different length (usually ISS data)    


    E = Es[0][:len(d0[0])-cut]+Eshift

    ds = xr.Dataset()
    try:
        arr_f = np.array([i[:len(d0[0])-cut] for i in MUs_f])
        da_f = xr.DataArray(data=arr_f[:,imin:imax],
                          coords=[np.arange(len(fl_in)), E[imin:imax]],
                          dims=['scan_num', 'energy']) 
        da_f.scan_num.attrs["files"] = fl_in
        ds['mu_fluo']  = deepcopy(da_f)

        try:
            arr_r = np.array([i[:len(d0[0])-cut] for i in MUs_r])
            da_r = xr.DataArray(data=arr_r[:,imin:imax],
                              coords=[np.arange(len(fl_in)), E[imin:imax]],
                              dims=['scan_num', 'energy'])
            da_r.scan_num.attrs["files"] = fl_in
            ds['mu_ref']   = deepcopy(da_r)
        except:
            pass


    except Exception as exc:
        print(exc)
        print('Unable to create dataset. Something is wrong')
        if plot and legend:
            ax1.legend(fontsize=8,loc='best',frameon=False)
            ax1.set_xlim(xlim)
            

    return ds



def deglitch(da_in,fl_in,glitches,plot=True):

    Is_new = []
    for i in da_in:
        Enew,Inew = i.energy.values.copy(), i.values.copy()
        for g in glitches:
            Etmp = [Enew[e] for e,s in enumerate(Enew) if (s < float(g.split(':')[0]) or s > float(g.split(':')[1])) ]
            Itmp = [Inew[e] for e,s in enumerate(Enew) if (s < float(g.split(':')[0]) or s > float(g.split(':')[1])) ]
            Enew,Inew = np.array(Etmp), np.array(Itmp)
        Is_new.append(Inew)
    Is_new = np.array(Is_new)      
    da_dg = xr.DataArray(data=Is_new,
                  coords=[np.arange(Is_new.shape[0]), Enew],
                  dims=['scan_num', 'energy']) 
    da_dg.scan_num.attrs["files"] = da_in.scan_num.attrs["files"]
    
    if plot:
        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(1,2,1)
        for e,i in enumerate(da_dg):
            i.plot.line('-',ms=1,ax=ax,label=fl_in[e][1])
        for e,i in enumerate(da_in):
            i.plot.line('--o',ms=1,ax=ax)
        ax.set_title(None)
        ax.set_xlabel('E (eV)')
        ax.set_ylabel('$\mu(E)$')
        ax.legend(fontsize=5,ncol=1,bbox_to_anchor=(1.1, 0.99))
        for g in glitches:
            ax.axvline(x=float(g.split(':')[0]),lw=0.2)
            ax.axvline(x=float(g.split(':')[1]),lw=0.2)
        plt.tight_layout()

    return da_dg




def normalize_and_flatten(da_in,e0=None,pre1=None,pre2=None,
                          nvict=2,norm1=None,norm2=None,
                          rbkg=1.0,kweight=1,kmin=2,kmax=10,dk=0.1,window='hanning',
                          ave_method='mean',xlim=None,
                          plot=True,figsize=(12,7),
                          show_edge_regions=True, show_raw=True,raw_plot_axes=[0.25, 0.25, 0.2, 0.45],
                          legend=False,show_std=True): 


    # create dataset
    ds = xr.Dataset()

    # first average all
    if ave_method == 'mean':
        ave = da_in.mean(axis=0)
    elif ave_method == 'median':
        ave = da_in.median(axis=0)
    else:
        ave = da_in.mean(axis=0)
        
    # pre_edge parameters
    if e0 is None:
        e0 = find_e0(ave.energy.values,ave.values)
        
    if pre1 is None:
        pre1 = -round(e0 - da_in.energy.values[1])
    if pre2 is None:
        pre2 = round(pre1/3)
        
    if norm2 is None:
        norm2 = round(da_in.energy.values[-2] - e0)
    if norm1 is None:
        norm1 = round(norm2/3)
            
    ds.attrs['e0'] = e0
    ds.attrs['pre1'] = pre1
    ds.attrs['pre2'] = pre2
    ds.attrs['nvict']= nvict
    ds.attrs['norm1']= norm1
    ds.attrs['norm2']= norm2


    # normalize and flatten each spectra individually
    mus = np.zeros((da_in.shape[0],da_in.shape[1]))
    norms = np.zeros((da_in.shape[0],da_in.shape[1]))
    flats = np.zeros((da_in.shape[0],da_in.shape[1]))
    for e,i in enumerate(da_in):
        group = Group(energy=da_in.energy.values, mu=da_in.isel(scan_num=e).values, filename=None)
        pre_edge(group, e0=e0, pre1=pre1, pre2=pre2, nvict=nvict, norm1=norm1, norm2=norm2, group=group)
        mus[e,:] = group.mu
        norms[e,:] = group.norm
        flats[e,:] = group.flat   
    da_mus = xr.DataArray(data=mus,
                      coords=[np.arange(norms.shape[0]), da_in.energy.values],
                      dims=['scan_num', 'energy'])
    ds['mus']  = deepcopy(da_mus)
    da_norms = xr.DataArray(data=norms,
                      coords=[np.arange(norms.shape[0]), da_in.energy.values],
                      dims=['scan_num', 'energy'])   
    ds['norms']  = deepcopy(da_norms)
    da_flats = xr.DataArray(data=flats,
                      coords=[np.arange(norms.shape[0]), da_in.energy.values],
                      dims=['scan_num', 'energy'])
    ds['flats']  = deepcopy(da_flats)


    


    
    # first average , then normalize
    if ave_method == 'mean':
        group_ave1 = Group(energy=da_mus.energy.values,mu=da_mus.mean(axis=0).values,filename=None)
    elif ave_method == 'median':
        group_ave1 = Group(energy=da_mus.energy.values,mu=da_mus.median(axis=0).values,filename=None)
    else:
        group_ave1 = Group(energy=da_mus.energy.values,mu=da_mus.mean(axis=0).values,filename=None)
    pre_edge(group_ave1, e0=e0, pre1=pre1, pre2=pre2, norm1=norm1, norm2=norm2, group=group_ave1)   

    ds['mu1'] = xr.DataArray(data=group_ave1.mu,
                      coords=[group_ave1.energy],
                      dims=['energy'])
    ds['flat1'] = xr.DataArray(data=group_ave1.flat,
                      coords=[group_ave1.energy],
                      dims=['energy'])
    ds['norm1'] = xr.DataArray(data=group_ave1.norm,
                      coords=[group_ave1.energy],
                      dims=['energy'])


    # first normalize , then average
    if ave_method == 'mean':
        da  = da_norms.mean(axis=0)
        group_ave2 = Group(energy=da.energy.values, mu=da.values, filename=None)
        pre_edge(group_ave2, e0=e0, pre1=pre1, pre2=pre2, norm1=norm1, norm2=norm2, group=group_ave2) 
    elif ave_method == 'median':
        da  = da_norms.median(axis=0)
        group_ave2 = Group(energy=da.energy.values, mu=da.values, filename=None)
        pre_edge(group_ave2, e0=e0, pre1=pre1, pre2=pre2, norm1=norm1, norm2=norm2, group=group_ave2) 
    else:
        da  = da_norms.mean(axis=0)
        group_ave2 = Group(energy=da.energy.values, mu=da.values, filename=None)
        pre_edge(group_ave2, e0=e0, pre1=pre1, pre2=pre2, norm1=norm1, norm2=norm2, group=group_ave2) 
     
    ds['mu2'] = xr.DataArray(data=group_ave2.mu,
                      coords=[group_ave2.energy],
                      dims=['energy'])
    ds['flat2'] = xr.DataArray(data=group_ave2.flat,
                      coords=[group_ave2.energy],
                      dims=['energy'])
    ds['norm2'] = xr.DataArray(data=group_ave2.norm,
                      coords=[group_ave2.energy],
                      dims=['energy'])   


    

    if plot:


        fig = plt.figure(figsize=figsize,dpi=96)

        ax = fig.add_subplot(1,2,1)
        for e,i in enumerate(da_flats):
            i.plot.line('-',ms=1,ax=ax)         
        if show_std:
            (da_flats.std(axis=0)-0.1).plot(ax=ax)
            
        ax.set_xlim(xlim)
        ax.set_title(None)
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Normalized and flattened $\mu(E)$')
        ax.axvline(x=e0+pre1,lw=0.2)
        ax.axvline(x=e0+pre2,lw=0.2)
        ax.axvline(x=norm1+e0,lw=0.2)
        ax.axvline(x=norm2+e0,lw=0.2)
        ax.axvline(x=e0,lw=0.2)  



        if show_edge_regions:
            ax = fig.add_axes([0.20, 0.2, 0.12, 0.3])
            ax.plot(group_ave1.energy,group_ave1.flat,'-r',lw=2)  
            ax.plot(group_ave2.energy,group_ave1.flat,'--b',lw=2)   
            ax.set_xlim([e0-20,e0])
            ax.set_ylim(top=0.5)
            ax.set_title('pre-edge')
            ax.set_yticklabels([])

            ax = fig.add_axes([0.34, 0.2, 0.12, 0.3])
            ax.plot(group_ave1.energy,group_ave1.flat,'-r',lw=2)   
            ax.plot(group_ave2.energy,group_ave1.flat,'--b',lw=2)   
            ax.set_xlim([e0,e0+20])
            ax.set_ylim(bottom=0.5)
            ax.set_title('post-edge')
            ax.set_yticklabels([])
        

        elif show_raw:
            ax = fig.add_axes(raw_plot_axes)

            for e,i in enumerate(da_mus):
                i.plot.line('-',ms=1,ax=ax,label=da_in.scan_num.attrs['files'][e][1].split('/')[-1])
            ax.set_title(None)
            ax.set_xlabel(None)
            ax.set_ylabel('$\mu(E)$')
            
            if legend:
                ax.legend(fontsize=6)     
            
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
        
        
        plt.tight_layout()
    
    
    try:
        autobk(group_ave1, rbkg=rbkg, kweight=kweight)  
        xftf(group_ave1, kmin=kmin, kmax=kmax, dk=dk, kwindow=window) 
        autobk(group_ave2, rbkg=rbkg, kweight=kweight)  
        xftf(group_ave2, kmin=kmin, kmax=kmax, dk=dk, kwindow=window) 

        ds.attrs['rbkg']= rbkg
        ds.attrs['kweight']= kweight
        ds.attrs['kmin']= kmin
        ds.attrs['kmax']= kmax
        ds.attrs['dk']= dk
        ds.attrs['window']= window    
        
        
        da = xr.DataArray(data=group_ave1.chir_mag,coords=[group_ave1.r],dims=['R'])
        ds['chir_mag1']  = deepcopy(da)        
        da = xr.DataArray(data=group_ave1.k*group_ave1.k*group_ave1.chi,coords=[group_ave1.k],dims=['k'])
        ds['k2chi1']  = deepcopy(da) 

        da = xr.DataArray(data=group_ave2.chir_mag,coords=[group_ave2.r],dims=['R'])
        ds['chir_mag2']  = deepcopy(da)        
        da = xr.DataArray(data=group_ave2.k*group_ave2.k*group_ave2.chi,coords=[group_ave2.k],dims=['k'])
        ds['k2chi2']  = deepcopy(da) 
        
        if plot:        
            ax = fig.add_subplot(1,2,2)
            ax.plot(group_ave1.r, group_ave1.chir_mag,'-r',lw=2)
            ax.plot(group_ave2.r, group_ave2.chir_mag,'--b',lw=2)
            ax.set_xlim([0,7])
            ax.set_xlabel('$\it{R}$ ($\AA$)')
            ax.set_ylabel('|$\chi$ ($\it{R}$)| ($\AA^{-3}$)')    
            ax.set_title('rbkg=%.2f, kmin=%.2f, kmax=%.2f \nkweight=%.2f, dk=%.2f, kwindow=%s'%(rbkg, kmin, kmax, kweight, dk, window),fontsize=9)

            ax = fig.add_axes([0.77, 0.60, 0.2, 0.3])
            ax.plot(group_ave1.k, group_ave1.k*group_ave1.k*group_ave1.chi,'-r')
            ax.plot(group_ave2.k, group_ave2.k*group_ave2.k*group_ave2.chi,'--b')
            ax.axvline(x=kmin,linestyle=':',color='k')
            ax.axvline(x=kmax,linestyle=':',color='k')
            ax.set_xlabel('$\it{k}$ ($\AA^{-1}$)')
            ax.set_ylabel('$\it{k^{2}}$ $\chi$ ($\it{k}$) ($\AA^{-2}$)')    

            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)        

            plt.tight_layout()
            
    except Exception as exc:
        print(exc)
        pass
    

    return ds


