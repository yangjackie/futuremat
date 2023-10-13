from core.phonon.anharmonic_score import *
import os
import tarfile
import shutil

root_dir = os.getcwd()
all_sigmas = []
temperatures = [200,300,400,500,600,700]
temperatures = [700]
for temp in temperatures:
    folder_name = 'disp_'+str(temp)+'K'
    folder_path = os.path.join(root_dir, folder_name)
    tar_files = [file_name for file_name in os.listdir(folder_path) if file_name.endswith(".tar.gz")][:40]
    path_holder = []

    for tar_count, tar_file in enumerate(tar_files):
        tar_file_path = os.path.join(folder_path, tar_file)

        print(f"{tar_count}/{len(tar_files)}, Processing {tar_file_path}... for temperature "+str(temp)+' K')
        with tarfile.open(tar_file_path, 'r') as tar:
            tar.extractall(path=folder_path)
            path_holder.append(os.path.join(folder_path, tar_file[:-7]))


    vasprun_xmls = [i+'/vasprun.xml' for i in path_holder if os.path.isfile(i+'/vasprun.xml')]

    scorer = AnharmonicScore(md_frames=vasprun_xmls, unit_cell_frame=root_dir+'/CONTCAR', ref_frame=root_dir+'/CONTCAR',
                             atoms=None,potim=1.0, mode_resolved=False,include_third_order=False,
                             force_constants=root_dir+'/force_constants.hdf5')
    sigma, time_stps = scorer.structural_sigma(return_trajectory=True)
    all_sigmas.append(sigma)

    #clear up the untarred directories
    for _path in path_holder:
        try:
            shutil.rmtree(os.path.join(_path))
        except:
            pass

