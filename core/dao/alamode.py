from core.dao.abstract_io import *
from core.models.crystal import Crystal


class AlamodeWriter(FileWriter):

    def __init__(self, crystal: Crystal):
        self.crystal = crystal
        self.unique_labels = self.unique_atoms(self.crystal)

    def write_alm_in_for_fitting_second_order(self, prefix='super_harm', input_name='ALM1.in', cutoff=[None]):

        f = open(input_name, 'w')
        f.write("&general\n")
        f.write(" PREFIX = " + prefix + '\n')  # prefix for the output force constants and xml file
        f.write(" MODE = opt\n")
        f.write(" NAT = " + str(self.crystal.total_num_atoms()) + '\n')  # total number of atoms
        f.write(" NKD = " + str(len(self.unique_labels)) + '\n')  # number of unique atoms
        f.write(" KD = ")
        for atom in self.unique_labels:
            f.write(str(atom) + '  ')
        f.write('\n')
        f.write("/\n\n")

        f.write("&interaction\n")
        f.write(" NORDER = 1\n")
        f.write("/\n\n")

        f.write("&cutoff\n")
        f.write(" *-* ")
        for i in cutoff:
            f.write(str(i) + " ")
        f.write("\n/\n\n")

        f.write("&optimize\n")
        f.write(" DFSET = DFSET_harmonic\n")
        f.write("/\n\n")

        self.write_lattice_vector_block(f)
        self.write_atom_block(f)
        f.close()

    def write_alm_in_for_fitting_higher_order_FCs(self, prefix='perovskite_300', second_fit_prefix='super_harm',
                                                  input_name='opt.in', norder=5, nbody=[2, 3, 3, 2, 2],
                                                  cutoff=[None, None, 12, 12, 12]):

        f = open(input_name, 'w')
        f.write("&general\n")
        f.write(" PREFIX = " + prefix + '\n')
        f.write(" MODE = opt\n")
        f.write(" NAT = " + str(self.crystal.total_num_atoms()) + '\n')  # total number of atoms
        f.write(" NKD = " + str(len(self.unique_labels)) + '\n')  # number of unique atoms
        f.write(" KD = ")
        for atom in self.unique_labels:
            f.write(str(atom) + '  ')
        f.write('\n')
        f.write("/\n\n")

        f.write("&interaction\n")
        f.write(" NORDER = " + str(norder) + '\n')
        f.write(" NBODY = ")
        for i in nbody:
            f.write(str(i) + " ")
        f.write('\n')
        f.write("/\n\n")

        f.write("&cutoff\n")
        f.write("*-* ")
        for i in cutoff:
            f.write(str(i) + " ")
        f.write('\n')
        f.write("/\n\n")

        f.write("&optimize\n")
        f.write(" FC2XML = " + second_fit_prefix + ".xml\n")
        f.write(" DFSET = DFSET\n")
        f.write(" LMODEL = enet\n")
        f.write(" CV = 0\n")
        f.write(" L1_RATIO = 1.0\n")
        f.write(" L1_ALPHA = 1e-6\n")
        f.write(" CV_MINALPHA = 2.0e-6\n")
        f.write(" CV_MAXALPHA = 0.02\n")
        f.write(" CV_NALPHA = 100\n")
        f.write(" NWRITE = 5000\n")
        f.write(" MAXITER = 1000000\n")
        f.write(" CONV_TOL = 1.0e-9\n")
        # f.write(" NSTART=400; NEND=500\n")
        f.write("/\n\n")

        self.write_lattice_vector_block(f)
        self.write_atom_block(f)
        f.close()

    def write_alm_in_for_scph(self, prefix='scph-run', input_name='scph.in', born_info=None,
                              high_order_fix_prefix='perovskite_300', mode='mesh', mesh_grid=[10, 10, 10]):

        f = open(input_name, 'w')
        f.write("&general\n")
        f.write(" PREFIX = " + prefix + '\n')
        f.write(" MODE = opt\n")
        f.write(" NKD = " + str(len(self.unique_labels)) + '\n')  # number of unique atoms
        f.write(" KD = ")
        for atom in self.unique_labels:
            f.write(str(atom) + '  ')
        f.write('\n')
        f.write(" FCSXML = " + high_order_fix_prefix + '.xml' + '\n')
        if born_info is not None:
            f.write('NONANALYTIC = 2; BORNINFO = ' + born_info + '\n')
        f.write('TMIN = 30; TMAX = 1000; DT = 10 \n')  # fix the temperature range that we are going to look at
        f.write("/\n\n")

        f.write("&scph\n")
        f.write(" SELF_OFFDIAG = 0\n")
        f.write(" MAXITER = 2000\n")
        f.write(" MIXALPHA = 0.2\n")
        f.write(" KMESH_INTERPOLATE = 2 2 2\n")
        f.write(" KMESH_SCPH = 2 2 2\n")
        f.write(" RESTART_SCPH = 0\n")
        f.write("/\n\n")

        f.write("&analysis\n")
        f.write(" PRINTMSD = 1")
        f.write("/\n\n")

        if mode == 'mesh':
            f.write("&kpoint\n")
            f.write(" 2\n")
            for i in mesh_grid:
                f.write(str(i) + ' ')
            f.write('\n' + "/\n\n")
        elif mode == 'line':
            raise NotImplementedError()

        self.write_lattice_vector_block(f)

        f.close()

    def write_atom_block(self, f):
        f.write("&position\n")
        all_atoms = self.crystal.all_atoms(sort=False)
        for iatom, atom in enumerate(all_atoms):
            f.write(str(self.unique_labels.index(atom.clean_label) + 1) + ' ')
            for i in range(3):
                f.write(' %19.16f' % atom.scaled_position[i])
            f.write('\n')
        f.write("/\n\n")

    def write_lattice_vector_block(self, f, factor=1.88973):
        f.write("&cell\n")
        f.write(str(factor) + "\n")
        latt_form = ' %21.16f'
        for row in range(3):
            f.write(' ')
            for column in range(3):
                f.write(latt_form % self.crystal.lattice.lattice_vectors.get(row, column))
            f.write('\n')
        f.write("/\n\n")

    def unique_atoms(self, crystal: Crystal) -> list:
        _all_atom_label = [i.clean_label for i in crystal.all_atoms(sort=False)]
        _unique_labels = []
        for l in _all_atom_label:
            if l not in _unique_labels:
                _unique_labels.append(l)
        return _unique_labels


class AlamodeReader(FileReader):

    def __init__(self, input_location=None, file_content=None):
        super(self.__class__, self).__init__(input_location=input_location,
                                             file_content=file_content)

    def read_kappa(self):
        f = open(self.input_location, 'r')
        kappa_dict = {}
        for line in f.readlines():
            if 'Temperature [K]' in line: pass
            _s = [float(s) for s in line.split()]
            temperature = _s[0]
            kappa_dict[temperature] = {'kxx': _s[1],
                                       'kxy': _s[2],
                                       'kxz': _s[3],
                                       'kyx': _s[4],
                                       'kyy': _s[5],
                                       'kyz': _s[6],
                                       'kzx': _s[7],
                                       'kzy': _s[8],
                                       'kzz': _s[9]}
        return kappa_dict
