import time

import dscribe
from dscribe.descriptors import SOAP
from dscribe.kernels import REMatchKernel
from sklearn.preprocessing import normalize
import argparse,os,pickle
from ase.io import read as ase_read
import numpy as np
import gc

def cosine_similarity(X,Y):
    return np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="ccontrol for soap analysis.")
    parser.add_argument("--compound", type=str, help="which compound to analyze")
    parser.add_argument("--build_kernel", action="store_true")
    parser.add_argument("--make_map", action="store_true")
    args = parser.parse_args()

    if ('F3' in args.compound) or ('F6' in args.compound):
        system = 'fluorides'
    elif ('Cl3' in args.compound) or ('Cl6' in args.compound):
        system = 'chlorides'
    elif ('Br3' in args.compound) or ('Br6' in args.compound):
        system = 'bromides'
    elif ('I3' in args.compound) or ('I6' in args.compound):
        system = 'iodides'

    pwd=os.getcwd()
    root_dir=system+'/dpv_'+args.compound+'/'



    if args.build_kernel:
        dft_traj_name = root_dir+'combined_dft.traj'
        mlff_traj_one_name = root_dir+'andersen_md_mace-omat-0-medium_run_1.traj'
        mlff_traj_two_name = root_dir+'andersen_md_mace-omat-0-medium_run_2.traj'

        #dft_traj = ase_read(dft_traj_name,':')

        trajectories = [ase_read(dft_traj_name,'::1')]
        for i in [3,4,5]:
           trajectories.append(ase_read(root_dir+'MD/vasprun_prod_'+str(i)+'.xml',index='::1'))

        trajectories.append((ase_read(mlff_traj_one_name,'::1')))
        trajectories.append((ase_read(mlff_traj_two_name, '::1')))

        counter = 0
        features = []

        r_cut = 5
        n_max = 7
        l_max = 6
        sigma = 0.1
        gamma = 2


        for traj in trajectories:
            for frame in traj:
                _atomic_numbers = frame.__dict__['arrays']['numbers']
                desc = SOAP(species=_atomic_numbers, r_cut=r_cut, n_max=n_max ,l_max=l_max, sigma=sigma, periodic=True, sparse=False)
                feature = desc.create(frame, centers=_atomic_numbers)
                feature = normalize(feature)
                features.append(feature)
                counter += 1
                print("descriptor done :", counter)

        if not os.path.exists(root_dir+'soap_kernel_long_DFT'):
            os.makedirs(root_dir+'soap_kernel_long_DFT')
        os.chdir(root_dir+'soap_kernel_long_DFT')

        f = open('soap_parameters.dat', 'w')
        f.write('r_cut:\t' + str(r_cut) + '\n')
        f.write('n_max:\t' + str(n_max) + '\n')
        f.write('l_max:\t' + str(l_max) + '\n')
        f.write('sigma:\t' + str(sigma) + '\n')
        f.write('gamma:\t' + str(gamma) + '\n')
        f.close()

        re = REMatchKernel(metric='linear',normalize_kernel='True',gamma=gamma)

        i=0
        import multiprocessing

        while len(features) > 1:
            start = time.time()
            _kernel = re.create(x=[features[0]], y=features[1:])
            del features[0]
            gc.collect()
            i+=1
            print("Built Kernel part " + str(i) + '   shape:' + str(np.shape(_kernel))+'   number of features left:'+str(len(features))+' loop time:'+str(time.time()-start))

            pickle.dump(_kernel, open('kernel_part_' + str(i) + '.bp', 'wb'))

        #for i in range(len(features) - 1):
        #    #_kernel = np.array([cosine_similarity(features[i][0], f[0]) for f in features[i + 1:]])
        #    _kernel = re.create(x=[features[i]], y=features[i+1:])
        #    print("Built Kernel part " + str(i)+'   shape:'+str(np.shape(_kernel)))
        #    pickle.dump(_kernel, open('kernel_part_' + str(i) + '.bp', 'wb'))
        print("kernel done")
        os.chdir(pwd)

    elif args.make_map:
        import glob
        filename = 'kernel_part_'
        all_kernel_files=glob.glob(filename+'*')

        kernel=np.zeros((len(all_kernel_files)+1,len(all_kernel_files)+1))

        for i in range(len(all_kernel_files)):
            data = pickle.load(open(filename+str(i)+'.bp', 'rb'))
            kernel[i][i+1:] = data
        kernel = kernel + kernel.T

        for i in range(len(all_kernel_files)+1):
            kernel[i][i]=1

        from sklearn.decomposition import PCA, KernelPCA
        import matplotlib.pyplot as plt

        kpca = KernelPCA(n_components=None, kernel="precomputed", fit_inverse_transform=False)
        X_kpca = kpca.fit_transform(kernel)
        fig, ax = plt.subplots()
        ax.scatter(-X_kpca[:, 0], X_kpca[:, 1])
        plt.tight_layout()
        plt.show()
