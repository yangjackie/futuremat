from twodPV.collect_data import *
import argparse
from pymatgen.io.vasp.outputs import *

def collect_dielectric_bulk(db):
    cwd = os.getcwd()
    base_dir = cwd + '/relax_Pm3m/'
    for i in range(len(A_site_list)):
        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:
                    system_name = a + b + c
                    uid = system_name + '3_pm3m'

                    print("Working on "+uid)
                    kvp = {}
                    data = {}
                    kvp['uid'] = uid

                    dir = os.path.join(base_dir, system_name + "_Pm3m")
                    os.chdir(dir)

                    try:
                        with zipfile.ZipFile('./phonon_G.zip') as z:
                            with open("./OUTCAR_ph", 'wb') as f:
                                f.write(z.read("OUTCAR"))
                        f.close()
                        z.close()
                    except:
                        pass

                    if os.path.isfile("./OUTCAR_ph"):
                        outcar = Outcar('./OUTCAR_ph')
                        outcar.read_lepsilon_ionic()
                        print('dielectric ionic tensor')
                        print(outcar.dielectric_ionic_tensor)
                        data['dielectric_ionic_tensor'] = outcar.dielectric_ionic_tensor
                        populate_db(db, None, kvp, data)
                        os.remove('./OUTCAR_ph')
                    os.chdir(cwd)

def collect_dielectric_2D(db):
    import tarfile,shutil
    cwd = os.getcwd()
    terminations = [['BO2'],['ABO','O2'],['AO3','B']]
    for orient_id, orient in enumerate(['100','110','111']):
        for term in terminations[orient_id]:
            base_dir = cwd + '/slab_'+orient+'_'+term+'_small'
            os.chdir(base_dir)

            for i in range(len(A_site_list)):
                for a in A_site_list[i]:
                    for b in B_site_list[i]:
                        for c in C_site_list[i]:
                            for thick in [3,5,7,9]:
                                system_name = a+b+c+'_'+str(thick)
                                to_continue=False
                                try:
                                    tf = tarfile.open(system_name + '.tar.gz')
                                    tf.extractall()
                                    #os.chdir(system_name)
                                    to_continue=True
                                except:
                                    logger.info(system_name + ' tar ball not working')
                                    continue

                                if to_continue:
                                    _this_dir = os.getcwd()
                                    collect_data = False
                                    try:
                                       os.chdir(system_name)
                                       collect_data = True
                                    except:
                                        pass

                                    if collect_data:
                                        data={}
                                        kvp={}
                                        uid = a+b+c + '3_' + str(orient) + "_" + str(term) + "_" + str(thick)
                                        kvp['uid'] = uid

                                        try:
                                            with zipfile.ZipFile('./phonon_G.zip') as z:
                                                with open("./OUTCAR_ph", 'wb') as f:
                                                    f.write(z.read("OUTCAR"))
                                            f.close()
                                            z.close()
                                        except:
                                            pass

                                        try:
                                            if os.path.isfile("./OUTCAR_ph"):
                                                outcar = Outcar('./OUTCAR_ph')
                                                outcar.read_lepsilon_ionic()

                                                logger.info(uid+'\t'+str(outcar.dielectric_ionic_tensor))
                                                data['dielectric_ionic_tensor'] = outcar.dielectric_ionic_tensor
                                                populate_db(db, None, kvp, data)
                                                os.remove('./OUTCAR_ph')
                                        except:
                                            pass

                                        os.chdir(_this_dir)

                                    try:
                                        shutil.rmtree(system_name)
                                    except:
                                        pass
                                    try:
                                        os.rmtree(system_name)
                                    except:
                                        pass

            os.chdir(cwd)

if __name__=="__main__":
    dbname = os.path.join(os.getcwd(), '2dpv.db')
    db = connect(dbname)
    logger = setup_logger(output_filename='data_collector_static_dielectric.log')
    collect_dielectric_2D(db)