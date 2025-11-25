from core.phonon.anharmonic_score import *
import matplotlib.pyplot as plt
import os

# get the DFT anharmonic scores
ref_frame = os.getcwd()+'/SPOSCAR'
dft_fc = os.getcwd()+'/force_constants.hdf5'

dft_sigma = []
for i in [1,2]:
    scorer = AnharmonicScore(md_frames=[os.getcwd()+'/MD/vasprun_prod_'+str(i)+'.xml'],
                            unit_cell_frame=ref_frame,
                            ref_frame=ref_frame,
                            atoms=None,
                            potim=1,
                            force_constants=dft_fc,
                            mode_resolved=False,
                            include_third_order=False)
    _dft_sigma, _dft_time_stps = scorer.structural_sigma(return_trajectory=True)
    dft_sigma += list(_dft_sigma)
dft_time_stps = range(len(dft_sigma))
plt.plot(dft_time_stps, dft_sigma,'b-')


#do this for the mlff-md
for i in range(10):
    mlff_md_frames = [os.getcwd()+'/trajectory-mace-mp-0b3-medium-'+str(i+1)+'.traj']
    scorer = AnharmonicScore(md_frames=mlff_md_frames,
                             unit_cell_frame=ref_frame,
                             ref_frame=ref_frame,
                             atoms=None,
                             potim=1,
                             force_constants=dft_fc,
                             mode_resolved=False,
                             include_third_order=False)
    mlff_sigma, mlff_time_stps = scorer.structural_sigma(return_trajectory=True)
    plt.plot(mlff_time_stps,mlff_sigma,'r-', alpha=0.5)


plt.show()