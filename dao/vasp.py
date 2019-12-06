import warnings

default_ionic_optimisation_set = {
    'SYSTEM': 'entdecker',
    'PREC': 'Normal',
    'EDIFF': 1E-5,
    'EDIFFG': -0.01,
    'IBRION': 2,
    'POTIM': 0.1,
    'ISIF': 0,
    'NSW': 800,
    'IALGO': 38,
    'ISYM': 0,
    'ADDGRID': '.TRUE.',
    'ISMEAR': 0,
    'SIGMA': 0.05,
    'LREAL': 'AUTO',
    'LWAVE': '.FALSE.',
    'LCHARG': '.FALSE.',
    'LVTOT': '.FALSE.',
    'LMAXMIX': 6,
    'AMIN': 0.01
}

default_static_calculation_set = {
    'SYSTEM': 'entdecker',
    'PREC': 'Normal',
    'INIWAV': 1,
    'ENCUT': 600,
    'EDIFF': 1E-5,
    'EDIFFG': -0.01,
    'IBRION': -1,
    'ISIF': 0,
    'NSW': 0,
    'IALGO': 38,
    'ISYM': 0,
    'ADDGRID': '.TRUE.',
    'ISMEAR': 0,
    'SIGMA': 0.05,
    'LREAL': 'AUTO',
    'LWAVE': '.FALSE.',
    'LCHARG': '.TRUE.',
    'LMAXMIX': 6,
    'NPAR': 28,
    'LORBIT': 11,
    'NELM': 500,
    'AMIN': 0.01,
    'LVTOT': '.TRUE.',
    'ISPIN': 2
}

class VaspWriter(object):

    def write_INCAR(self, filename='INCAR',
                    default_options=default_ionic_optimisation_set,
                    **kwargs):
        default_options.update(kwargs)

        incar = open(filename, 'w')
        incar.write("SYSTEM=" + str(default_options['SYSTEM']) + '\n')

        keylist = default_options.keys()
        keylist = list(sorted(keylist))

        #if ('KPAR' not in [k.upper() for k in keylist]) or ('NPAR' not in [k.upper() for k in keylist]):
        #    warnings.warn("KPAR or NPAR not set for specific architecture, calculation might be very slow!")

        for key in keylist:
            if key not in ['frame', 'kwargs', 'self', 'SYSTEM', 'filename']:
                incar.write(str(key) + '=' + str(default_options[key]) + '\n')

        incar.close()