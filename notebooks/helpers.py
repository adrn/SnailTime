import astropy.coordinates as coord
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic, binned_statistic_2d

vcirc = 229 * u.km/u.s
with coord.galactocentric_frame_defaults.set('v4.0'):
    galcen_frame = coord.Galactocentric()

solar_action_units = [0.0083016412, -2.0345427, 0.00043789795] * u.kpc**2 / u.Myr
    

def plot_spiral(
    x1, x2, 
    arr=None, 
    bins=None, 
    smooth=1.,        
    arr_statistic='median',     
    shuffle_subtract=False,
    shuffle_subtract_smooth=4,
    ax=None, 
    mesh=None,
    pcolor_kw=None, 
    xlabel=None, 
    ylabel=None,
    colorbar=True, 
    colorbar_kw=None, 
    colorbar_label=''
):
    """
    Parameters
    ----------
    x1 :
    x2 : 
    arr : array-like (optional)
        By default, when not specified, this produces a 2D array
        of counts, i.e. a 2D histogram. When specified, this uses 
        `binned_statistic_2d` to compute the statistic specified
        by `arr_statistic` at each bin.
    bins : array-like, int (optional)
        The bins passed to `numpy.histogram2d` or 
        `scipy.stats.binned_statistic_2d`.
    smooth : numeric (optional)
        Gaussian smoothing.
    arr_statistic : str, callable (optional)
        When specified with `arr`, this is the statistic to compute
        on the values of `arr` in each pixel.
    """
    if ax is None:
        ax = plt.gca()
    fig = ax.figure
        
    if pcolor_kw is None:
        pcolor_kw = dict()
        
    if (x1.unit.physical_type == 'speed' 
            and x2.unit.physical_type == 'length'):
        default_xlabel = f'$v_z$ [{x1.unit:latex_inline}]'
        default_ylabel = f'$z$ [{x2.unit:latex_inline}]'
        
        default_bins = (
            (np.arange(-75, 75+1e-3, 1.) * u.km/u.s).to_value(x1.unit),
            (np.arange(-1.5, 1.5+1e-3, 25 / 1e3)*u.kpc).to_value(x2.unit)
        )
        
        if shuffle_subtract:
            raise ValueError('shuffle_subtract is only supported for Jz, thetaz input')
        
    elif (x1.unit.physical_type == 'kinematic viscosity'
             and x2.unit.physical_type == 'angle'):
        default_xlabel = r'$\sqrt{J_z}\,\cos\theta_z$ ' + f'[{np.sqrt(x1).unit:latex_inline}]'
        default_ylabel = r'$\sqrt{J_z}\,\sin\theta_z$ ' + f'[{np.sqrt(x1).unit:latex_inline}]'
        
        if shuffle_subtract:
            if arr is not None:
                raise ValueError('shuffle_subtract is only supported for histograms')
                
            # TODO: allow as input?
            rng = np.random.default_rng()
            
            # Make fake phases:
            shuffled_th_z = x2.copy()
            rng.shuffle(shuffled_th_z)
            
            shuff_x1 = np.sqrt(x1) * np.cos(shuffled_th_z)
            shuff_x2 = np.sqrt(x1) * np.sin(shuffled_th_z)
        
        _x1 = np.sqrt(x1) * np.cos(x2)
        _x2 = np.sqrt(x1) * np.sin(x2)
        x1 = _x1
        x2 = _x2
        
        default_bins = (
            np.linspace(-np.sqrt(75*1.5), np.sqrt(75*1.5), 151) * np.sqrt(1 * u.km/u.s * u.kpc)
        ).to_value(x1.unit)
        
    else:
        raise ValueError("Unsupported input.")
        
    if bins is None:
        bins = default_bins
        
    if xlabel is None:
        xlabel = default_xlabel
    
    if ylabel is None:
        ylabel = default_ylabel
        
    if arr is None:
        H, xe, ye = np.histogram2d(
            x1.value,
            x2.value,
            bins=bins
        )
    
    else:
        stat = binned_statistic_2d(
            x1.value,
            x2.value,
            arr,
            statistic=arr_statistic,
            bins=bins
        )
        H = stat.statistic
        xe = stat.x_edge
        ye = stat.y_edge
    
    if smooth is not None:
        kernel = Gaussian2DKernel(x_stddev=smooth)
        H = convolve(H, kernel)
        
    if shuffle_subtract:
        H_shuff, xe, ye = np.histogram2d(
            shuff_x1.value,
            shuff_x2.value,
            bins=bins
        )
        H_shuff = convolve(H, Gaussian2DKernel(x_stddev=shuffle_subtract_smooth))
        H = (H - H_shuff) / H_shuff
    
    if 'norm' not in pcolor_kw:
        pcolor_kw.setdefault('vmin', np.nanpercentile(H.ravel(), 5))
        pcolor_kw.setdefault('vmax', np.nanpercentile(H.ravel(), 95))
    
    if mesh is not None:
        mesh.set_array(H.T)
    else:
        mesh = ax.pcolormesh(xe, ye, H.T, **pcolor_kw)
    
    if colorbar:
        if colorbar_kw is None:
            colorbar_kw = dict()
        colorbar_kw.setdefault('aspect', 30)
            
        cb = fig.colorbar(mesh, ax=ax, **colorbar_kw)
        cb.set_label(colorbar_label)
        
    if xlabel:
        ax.set_xlabel(xlabel)
        
    if ylabel:
        ax.set_ylabel(ylabel)
        
    return ax, mesh
