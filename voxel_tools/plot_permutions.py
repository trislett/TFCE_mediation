#!/usr/bin/python

import os
import sys
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

# from: http://www.jesshamrick.com/2012/09/03/saving-figures-from-pyplot/
def save(path, ext='png', close=True, verbose=True):
    """Save a figure from pyplot.
    Parameters
    ----------
    path : string
        The path (and filename, without the extension) to save the
        figure to.
    ext : string (default='png')
        The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
    close : boolean (default=True)
        Whether to close the figure after saving.  If you want to save
        the figure multiple times (e.g., to multiple formats), you
        should NOT close it in between saves or you will have to
        re-plot it.
    verbose : boolean (default=True)
        Whether to print information about when and where the image
        has been saved.
    """
    
    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # The final path to save to
    savepath = os.path.join(directory, filename)

    if verbose:
        print("Saving figure to '%s'..." % savepath),

    # Actually save the figure
    plt.savefig(savepath)
    
    # Close it
    if close:
        plt.close()

    if verbose:
        print("Done")

cmdargs = str(sys.argv)
arg_permutations = str(sys.argv[1])
#arg_permutations = str('perm_Tstat_TFCE_max_voxel_lh.csv')
perm_tfce_max = np.genfromtxt(arg_permutations, delimiter=',')
p_array=np.zeros(perm_tfce_max.shape)
sorted_perm_tfce_max=sorted(perm_tfce_max, reverse=True)
num_perm=perm_tfce_max.shape[0]
perm_tfce_mean = perm_tfce_max.mean()
perm_tfce_std = perm_tfce_max.std()
perm_tfce_max_val = int(sorted_perm_tfce_max[0])
perm_tfce_min_val = int(sorted_perm_tfce_max[(num_perm-1)])

for j in xrange(num_perm):
	p_array[j] = 1 - np.true_divide(j,num_perm)

sig=int(num_perm*0.05)
firstquater=sorted_perm_tfce_max[int(num_perm*0.75)]
median=sorted_perm_tfce_max[int(num_perm*0.50)]
thirdquater=sorted_perm_tfce_max[int(num_perm*0.25)]
sig_tfce=sorted_perm_tfce_max[sig]
pl.hist(perm_tfce_max, 100, range=[0,perm_tfce_max_val], label='Max TFCE scores')
ylim = pl.ylim()

pl.plot([sig_tfce,sig_tfce], ylim, '--g', linewidth=3,label='P[FWE]=0.05')
pl.text((sig_tfce*1.4),(ylim[1]*.5), r"$\mu=%0.2f,\ \sigma=%0.2f$" "\n" r"$Critical\ TFCE\ value=%0.0f$" "\n" r"$[%d,\ %d,\ %d,\ %d,\ %d]$" % (perm_tfce_mean,perm_tfce_std,sig_tfce,perm_tfce_min_val,firstquater, median, thirdquater, perm_tfce_max_val), size='medium')
pl.ylim(ylim)
pl.legend()
pl.xlabel('Permutation scores')
save("%s.hist" % arg_permutations, ext="png", close=False, verbose=True)
pl.show()



