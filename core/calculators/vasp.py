import os
import logging

logger = logging.getLogger("futuremat.core.calculators.vasp")

from core.calculators.abstract_calculator import Calculator
from core.dao.vasp import *

# Parameters that can be set in INCAR. The values which are None
# are not written and default parameters of VASP are used for them.

float_keys = [
    'aexx',  # Fraction of exact/DFT exchange
    'aggac',  # Fraction of gradient correction to correlation
    'aggax',  # Fraction of gradient correction to exchange
    'aldac',  # Fraction of LDA correlation energy
    'amin',  #
    'amix',  #
    'amix_mag',  #
    'andersen_prob',  # probability for colliding with the thermal bath to maintain constant temperatrue in thermostat.
    'bmix',  # tags for mixing
    'bmix_mag',  #
    'deper',  # relative stopping criterion for optimization of eigenvalue
    'ebreak',  # absolute stopping criterion for optimization of eigenvalues (EDIFF/N-BANDS/4)
    'emax',  # energy-range for DOSCAR file
    'emin',  #
    'enaug',  # Density cutoff
    'encut',  # Planewave cutoff
    'encutfock',  # FFT grid in the HF related routines
    'encutgw',
    'hfscreen',  # attribute to change from PBE0 to HSE
    'potim',  # time-step for ion-motion (fs)
    'nelect',  # total number of electronscd
    'param1',  # Exchange parameter
    'param2',  # Exchange parameter
    'pomass',  # mass of ions in am
    'sigma',  # broadening in eV
    'time',  # special control tag
    'weimin',  # maximum weight for a band to be considered empty
    'zab_vdw',  # vdW-DF parameter
    'zval',  # ionic valence
    # The next keywords pertain to the VTST add-ons from Graeme Henkelman's group at UT Austin
    'jacobian',  # Weight of lattice to atomic motion
    'ddr',  # (DdR) dimer separation
    'drotmax',  # (DRotMax) number of rotation steps per translation step
    'dfnmin',  # (DFNMin) rotational force below which dimer is not rotated
    'dfnmax',  # (DFNMax) rotational force below which dimer rotation stops
    'stol',  # convergence ratio for minimum eigenvalue
    'sdr',  # finite difference for setting up Lanczos matrix and step size when translating
    'maxmove',  # Max step for translation for IOPT > 0
    'invcurve',  # Initial curvature for LBFGS (IOPT = 1)
    'timestep',  # Dynamical timestep for IOPT = 3 and IOPT = 7
    'sdalpha',  # Ratio between force and step size for IOPT = 4
    # The next keywords pertain to IOPT = 7 (i.e. FIRE)
    'ftimemax',  # Max time step
    'ftimedec',  # Factor to dec. dt
    'ftimeinc',  # Factor to inc. dt
    'falpha',  # Parameter for velocity damping
    'falphadec',  # Factor to dec. alpha
]

exp_keys = [
    'ediff',  # stopping-criterion for electronic upd.
    'ediffg',  # stopping-criterion for ionic upd.
    'symprec',  # precession in symmetry routines
    # The next keywords pertain to the VTST add-ons from Graeme Henkelman's group at UT Austin
    'fdstep',  # Finite diference step for IOPT = 1 or 2
]

string_keys = [
    'algo',  # algorithm: Normal (Davidson) | Fast | Very_Fast (RMM-DIIS)
    'gga',  # xc-type: PW PB LM or 91
    'prec',  # Precission of calculation (Low, Normal, Accurate)
    'system',  # name of System
    'tebeg',  #
    'teend',  # temperature during run
    'precfock'
]

int_keys = [
    'ialgo',  # algorithm: use only 8 (CG) or 48 (RMM-DIIS)
    'ibrion',  # ionic relaxation: 0-MD 1-quasi-New 2-CG
    'icharg',  # charge: 0-WAVECAR 1-CHGCAR 2-atom 10-const
    'idipol',  # monopol/dipol and quadropole corrections
    'iniwav',  # initial electr wf. : 0-lowe 1-rand
    'isif',  # calculate stress and what to relax
    'ismear',  # part. occupancies: -5 Blochl -4-tet -1-fermi 0-gaus >0 MP
    'ispin',  # spin-polarized calculation
    'istart',  # startjob: 0-new 1-cont 2-samecut
    'isym',  # symmetry: 0-nonsym 1-usesym 2-usePAWsym
    'iwavpr',  # prediction of wf.: 0-non 1-charg 2-wave 3-comb
    'ivdw',  # dispersion correction
    'ldauprint',  # 0-silent, 1-occ. matrix written to OUTCAR, 2-1+pot. matrix written
    'ldautype',  # L(S)DA+U: 1-Liechtenstein 2-Dudarev 4-Liechtenstein(LDAU)
    'lmaxmix',  #
    'lorbit',  # create PROOUT
    'maxmix',  #
    'mdalgo',
    'ngx',  # FFT mesh for wavefunctions, x
    'ngxf',  # FFT mesh for charges x
    'ngy',  # FFT mesh for wavefunctions, y
    'ngyf',  # FFT mesh for charges y
    'ngz',  # FFT mesh for wavefunctions, z
    'ngzf',  # FFT mesh for charges z
    'nbands',  # Number of bands
    'nblk',  # blocking for some BLAS calls (Sec. 6.5)
    'ncore',
    'nbmod',  # specifies mode for partial charge calculation
    'nelm',  # nr. of electronic steps (default 60)
    'nelmdl',  # nr. of initial electronic steps
    'nelmin',
    'nedos',
    'nfree',  # number of steps per DOF when calculting Hessian using finitite differences
    'nkred',  # define sub grid of q-points for HF with nkredx=nkredy=nkredz
    'nkredx',  # define sub grid of q-points in x direction for HF
    'nkredy',  # define sub grid of q-points in y direction for HF
    'nkredz',  # define sub grid of q-points in z direction for HF
    'npar',  # parallelization over bands
    'kpar',  # parallelization over k-point
    'nsim',  # evaluate NSIM bands simultaneously if using RMM-DIIS
    'nsw',  # number of steps for ionic upd.
    'nupdown',  # fix spin moment to specified value
    'nwrite',  # verbosity write-flag (how much is written)
    'smass',  # Nose mass-parameter (am)
    'vdwgr',  # extra keyword for Andris program
    'vdwrn',  # extra keyword for Andris program
    'voskown',  # use Vosko, Wilk, Nusair interpolation
    # The next keywords pertain to the VTST add-ons from Graeme Henkelman's group at UT Austin
    'ichain',  # Flag for controlling which method is being used (0=NEB, 1=DynMat, 2=Dimer, 3=Lanczos)
    # if ichain > 3, then both IBRION and POTIM are automatically set in the INCAR file
    'iopt',  # Controls which optimizer to use.  for iopt > 0, ibrion = 3 and potim = 0.0
    'snl',  # Maximum dimentionality of the Lanczos matrix
    'lbfgsmem',  # Steps saved for inverse Hessian for IOPT = 1 (LBFGS)
    'fnmin',  # Max iter. before adjusting dt and alpha for IOPT = 7 (FIRE)
    'omegamax'
]

bool_keys = [
    'addgrid',  # finer grid for augmentation charge density
    'laechg',  # write AECCAR0/AECCAR1/AECCAR2
    'lasph',  # non-spherical contributions to XC energy (and pot for VASP.5.X)
    'lasync',  # overlap communcation with calculations
    'lcharg',  #
    'lcorr',  # Harris-correction to forces
    'lcalcpol',
    'lcalceps',
    'ldau',  # L(S)DA+U
    'ldiag',  # algorithm: perform sub space rotation
    'ldipol',  # potential correction mode
    'lelf',  # create ELFCAR
    'lepsilon',
    'lhfcalc',  # switch to turn on Hartree Fock calculations
    'loptics',  # calculate the frequency dependent dielectric matrix
    'lpard',  # evaluate partial (band and/or k-point) decomposed charge density
    'lplane',  # parallelisation over the FFT grid
    'lscalapack',  # switch off scaLAPACK
    'lscalu',  # switch of LU decomposition
    'lsepb',  # write out partial charge of each band seperately?
    'lsepk',  # write out partial charge of each k-point seperately?
    'lsorbit',  # whether to include spin-orbit coupling?
    'lthomas',  #
    'luse_vdw',  # Invoke vdW-DF implementation by Klimes et. al
    'lvhar',  # write Hartree potential to LOCPOT (vasp 5.x)
    'lvtot',  # create WAVECAR/CHGCAR/LOCPOT
    'lwave',  #
    # The next keywords pertain to the VTST add-ons from Graeme Henkelman's group at UT Austin
    'lclimb',  # Turn on CI-NEB
    'ltangentold',  # Old central difference tangent
    'ldneb',  # Turn on modified double nudging
    'lnebcell',  # Turn on SS-NEB
    'lglobal',  # Optmizize NEB globally for LBFGS (IOPT = 1)
    'llineopt',  # Use force based line minimizer for translation (IOPT = 1)
    'lrpa',  # whether exchange-correlation kernel should be used
]

list_keys = [
    'dipol',  # center of cell for dipol
    'eint',  # energy range to calculate partial charge for
    'ferwe',  # Fixed band occupation (spin-paired)
    'ferdo',  # Fixed band occupation (spin-plarized)
    'iband',  # bands to calculate partial charge for
    'magmom',  # initial magnetic moments
    'kpuse',  # k-point to calculate partial charge for
    'ropt',  # number of grid points for non-local proj in real space
    'rwigs',  # Wigner-Seitz radii
    'ldauu',  # ldau parameters, has potential to redundant w.r.t. dict
    'ldaul',  # key 'ldau_luj', but 'ldau_luj' can't be read direct from
    'ldauj',  # the INCAR (since it needs to know information about atomic
    # species. In case of conflict 'ldau_luj' gets written out
    # when a calculation is set up
]

special_keys = [
    'lreal',  # non-local projectors in real space
]

dict_keys = [
    'ldau_luj',  # dictionary with L(S)DA+U parameters, e.g. {'Fe':{'L':2, 'U':4.0, 'J':0.9}, ...}
]

keys = [
    # 'NBLOCK' and KBLOCK       inner block; outer block
    # 'NPACO' and APACO         distance and nr. of slots for P.C.
    # 'WEIMIN, EBREAK, DEPER    special control tags
]

mlff_keys = [
    'ML_AFILT2', #This tag sets the filtering parameter for the angular filtering for ML_IAFILT2 in the machine learning force field method.
    'ML_CDOUB', # This tag controls the criterion for "enforced" DFT calculations within the machine learning force field method. If at any time, the estimated force errors are ML_CDOUB times larger than the Bayesian threshold (i.e. "critically" high), a first principles calculation is performed and a new force field is immediately generated (even if the counter for sampling is below the minimum amount of sampled structures ML_NMDINT).
    'ML_CSIG', #  Parameter used in the automatic determination of threshold ML_CTIFOR for Bayesian error estimation in the machine learning force field method.
    'ML_CSLOPE', # Parameter used in the automatic determination of threshold for Bayesian error estimation in the machine learning force field method.
    'ML_CTIFOR', # This flag sets the threshold for the Bayesian error estimation on the force within the machine learning force field method.
    'ML_CX', #The parameter determines to which value the threshold (ML_CTIFOR) is updated within the machine learning force field methods.
    'ML_EATOM_REF', #Reference total energies of isolated atoms used in the machine learning force field method.
    'ML_EPS_LOW', #Threshold for the CUR algorithm used in the sparsification of local reference configurations within the machine learning force fields.
    'ML_EPS_REG', #Initial value for the threshold of the eigenvalues of the covariance matrix in the evidence approximation.
    'ML_IAFILT2', #This tag specifies the type of angular filtering used in the machine learning force field method.
    'ML_IALGO_LINREG', #This tag determines the algorithm that is employed to solve the system of linear equations in the ridge regression method for machine learning.
    'ML_ICOUPLE', #This tag specifies the atoms where the coupling parameter is introduced to calculate the chemical potential within the machine learning force field method.
    'ML_ICRITERIA', #Decides whether (ML_ICRITERIA>0) or how the Bayesian error threshold (ML_CTIFOR) is updated within the machine learning force field method. ML_CTIFOR determines whether a first principles calculations is performed.
    'ML_IERR', #Calculation and output frequency of Bayesian error estimate.
    'ML_IREG', #This tag specifies whether the regularization parameters are kept constant or not in the machine learning force field method.
    'ML_ISCALE_TOTEN', #This tag specifies how to scale the energy data in the machine learning force field method.
    'ML_ISTART', #This tag selects the mode of operation (e.g. start from scratch, prediction-only,...) of the machine learning force fields method.
    'ML_IWEIGHT', #This tag controls which procedure is used for normalizing and weighting the energies, forces and stresses in the machine learning force field method.
    'ML_LAFILT2', #This tag specifies whether angular filtering is applied or not within the machine learning force field method.
    'ML_LBASIS_DISCARD', #Controls whether calculation is continued or stopped after the maximum number of local reference configurations ML_MB for a given species is reached.
    'ML_LCOUPLE', #his tag specifies whether thermodynamic integration is activated in order to calculate the chemical potentials within the machine learning force field method.
    'ML_LEATOM', #This term specifies whether the total atomic energy is written out or not.
    'ML_LERR', #Decides whether the Bayesian error estimates are calculated and written out or not.
    'ML_LFAST', #This tag switches on the descriptors for refitting in the fast execution mode within machine learning force fields.
    'ML_LHEAT', #This tag specifies whether the heat flux is calculated or not in the machine learning force field method.
    'ML_LMAX2', #This tag specifies the maximum angular momentum quantum number  of spherical harmonics used to expand atomic distributions within the machine learning force field method.
    'ML_LMLFF', #Main control tag which enables/disables the use of machine learning force fields.
    'ML_LSPARSDES', #This tag specifies whether angular descriptor sparsification is enabled within the machine learning force field method.
    'ML_MB', # This tag sets the maximum number of local reference configurations (i.e. basis functions in the kernel) in the machine learning force field method.
    'ML_MCONF', #This tag sets the maximum number of structures stored in memory which are used for training in the machine learning force field method.
    'ML_MCONF_NEW', #This tag sets the number of configurations that are stored temporarily as candidates for the training data in the machine learning force field method.
    'ML_MHIS', #This tag sets the number of estimated errors stored in memory to determine the threshold for the Bayesian error in the machine learning force field method for ML_ICRITERIA=1. For ML_ICRITERIA=2, the history length is 50 x ML_MHIS (or hard coded to 400).
    'ML_MODE', #String based tag selecting operation mode for machine learning force fields.
    'ML_MRB1', #Number of radial basis function
    'ML_MRB2', #Number of radial basis function
    'ML_NATOM_COUPLED', #This tag specifies the number of atoms for which a coupling parameter is introduced to calculate the chemical potential within the machine learning force field method.
    'ML_NHYP', #his tag specifies the polynomial power zeta of the kernel within the machine learning force field methos
    'ML_NMDINT', #Tag to control the minimum interval to get training samples in the machine learning force field method.
    'ML_NRANK_SPARSDES', #This tag sets the number of highest eigenvalues to which the correlation is measured within the angular descriptor sparsification (within the machine learning force field method).
    'ML_OUTBLOCK', #Output distance in number of steps of the molecular-dynamics results for ML_ISTART=2 within the machine learning force fields.
    'ML_OUTPUT_MODE', #This tag decides the output verbosity of the molecular-dynamics calculation using machine learning.
    'ML_RCOUPLE', #This tag specifies the value of the coupling parameter for the calculation of the chemical potential within the machine learning force field method.
    'ML_RCUT1', #This flag sets the cutoff radius for radial function
    'ML_RCUT2', #This flag sets the cutoff radius for radial function
    'ML_RDES_SPARSDES', #This tag sets the ratio of the selected to the total number of descriptors within the angular descriptor sparsification (within the machine learning force field method).
    'ML_SCLC_CTIFOR', # Sets fraction by which the Bayesian threshold for the maximum forces is lowered in the selection of local reference calculations.
    'ML_SIGV0', #This flag sets the initial reversed and squared noise parameter
    'ML_SIGW0', #This flag sets the initial reversed and squared precision parameter
    'ML_SION1', #Gaussian width
    'ML_SION2', #Gaussian width
    'ML_W1', #This tag defines the weight  for the radial (and angular) descriptor within the machine learning force field method .
    'ML_WTIFOR', #This tag sets the weight for the scaling of the forces in the training data within the machine learning force field method.
    'ML_WTOTEN', #This tag sets the weight for the scaling of the total energy in the training data within the machine learning force field method.
    'ML_WTSIF' #This tag sets the weight for the scaling of the stress in the training data within the machine learning force field method.
]

all_incar_keys = dict_keys + special_keys + list_keys + bool_keys + int_keys + string_keys + exp_keys + \
                 float_keys + [str(m).lower() for m in mlff_keys]


class Vasp(Calculator):

    def __init__(self, **kwargs):
        self.writer = VaspWriter()
        # self.reader = VaspReader()
        self.set_incar_params(**kwargs)
        self.set_executable(**kwargs)
        self.set_mp_grid_density(**kwargs)
        self.completed = False
        self.self_consistency_error = False

        try:
            self.magnetic = kwargs['magnetic']
        except KeyError:
            self.magnetic = False

        try:
            self.gpu_run = kwargs['gpu_run']
        except KeyError:
            self.gpu_run = False

        try:
            self.use_gw = kwargs['use_gw']
        except KeyError:
            self.use_gw = False

        try:
            self.clean_after_success = kwargs['clean_after_success']
        except KeyError:
            self.clean_after_success = True

        try:
            self.gamma_centered = kwargs['Gamma_centered']
        except KeyError:
            self.gamma_centered = False

        try:
            self.kpoint_str = kwargs['KPOINT_string']
        except KeyError:
            self.kpoint_str = None

        try:
            self.write_poscar = kwargs['write_poscar']
        except KeyError:
            self.write_poscar = True

    def set_mp_grid_density(self, **kwargs):
        self.mp_grid_density = None
        self.MP_points = None
        if 'mp_grid_density' in kwargs.keys():
            self.mp_grid_density = kwargs['mp_grid_density']
        else:
            if 'MP_points' in kwargs.keys():
                self.MP_points = kwargs['MP_points']
            else:
                self.mp_grid_density = 0.04

    def set_incar_params(self, **kwargs):
        self.incar_params = {}
        for key in kwargs.keys():
            if key.lower() in all_incar_keys:
                self.incar_params[key.lower()] = kwargs[key]

    def set_executable(self, **kwargs):
        if 'executable' not in kwargs.keys():
            self.executable = 'vasp_std'
        else:
            # assert kwargs['executable'] in ['vasp_std', 'vasp_gam', 'vasp_std-xy', 'vasp_std-xz','vasp_std-yz']
            self.executable = kwargs['executable']

    def _update_executable(self):
        # update the vasp executable depending on the k-Point settings
        if (self.crystal.gamma_only is True) and ('tst' not in self.executable):
            logger.info("Using gamma only version VASP for calculations @ Gamma point only.")
            if self.executable != 'vasp_gam-xy':
                self.executable = 'vasp_gam'
        else:
            if self.executable == 'vasp_gam':
                self.executable = 'vasp_std'
        logger.info("VASP calculation to be executed with the following binary: " + str(self.executable))

    def setup(self):
        """
        This methods set up a VASP calculation for a given crystal, including writing out the POTCAR,
        INCAR and KPOINT files
        """
        logger.info('Setting up VASP calculation, write input file ...')

        if self.write_poscar:
            self.writer.write_structure(self.crystal, filename='POSCAR', magnetic=self.magnetic)
            logger.info('POSCAR written')
        else:
            logger.info("Skip writing POSCAR, use existing POSCAR (temp hack for MD runs")

        self.writer.write_potcar(self.crystal, sort=False, unique=True, magnetic=self.magnetic, use_GW=self.use_gw)
        logger.info('POTCAR written')

        # Check if there is already a KPOINTS file exists, for example, use the setting from
        # a previous calculation or database record, then skip writing our own KPOINTS file
        # An extra check will be done to see if wwe will be running Gamma only calculation
        self._setup_kpoints()
        self._update_executable()

        self.writer.write_INCAR('INCAR', default_options=self.incar_params)
        logger.info("INCAR written")

    def _setup_kpoints(self):
        if self.kpoint_str is None:
            if not os.path.isfile('./KPOINTS'):
                logger.info('No existing KPOINTS file, autogenerate Monkhorst-Pack K-point with density of ' + str(
                    self.mp_grid_density) + ' A^-1.')
                self.writer.write_KPOINTS(self.crystal, grid=self.mp_grid_density, K_points=self.MP_points,
                                          gamma_centered=self.gamma_centered)
            else:
                logger.info('Found existing KPOINTS, using previous set up')
                f = open('./KPOINTS', 'r')
                for l in f.readlines():
                    if '1 1 1' in l:
                        self.crystal.gamma_only = True
        else:
            self.writer.write_KPOINTS(self.crystal, kpoint_str=self.kpoint_str)

    def tear_down(self):
        """
        Clean up the calculation folder after VASP finishes execution
        """
        logger.info("Clean up directory after VASP executed successfully.")
        files = ['CHG', 'CHGCAR', 'EIGENVAL', 'IBZKPT', 'PCDAT', 'POTCAR', 'WAVECAR', 'LOCPOT',
                 'node_info', "WAVECAR", "WAVEDER", 'DOSCAR', 'PROCAR', 'REPORT']
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass

    def run(self):
        logger.info("Start executing VASP")

        if self.gpu_run:
            logger.info("Choose to run with GPU, reset executable to vasp-gpu")
            if self.crystal.gamma_only is True:
                cmd = 'mpirun -np $PBS_NGPUS --map-by ppr:1:numa vasp_gam-gpu'
            else:
                cmd = 'mpirun -np $PBS_NGPUS --map-by ppr:1:numa vasp_std-gpu'
        else:
            cmd = 'mpirun ' + self.executable

        exitcode = os.system('%s > %s' % (cmd, 'vasp.log'))
        if exitcode != 0:
            raise RuntimeError('Vasp exited with exit code: %d.  ' % exitcode)

    def check_convergence(self, outcar=None):
        """Method that checks whether a calculation has converged. Adapted from ASE."""
        ibrion = None
        nsw = None
        opt_iterations = None
        ediff = None
        # First check electronic convergence
        if outcar is None:
            outcar = './OUTCAR'
        outcar = open(outcar, 'r')
        for line in outcar.readlines():
            if line.rfind('Call to ZHEGV failed') > -1:
                self.self_consistency_error = True
                self.completed = False
                logger.info("VASP crashed out due to error in SCF cycles? " + str(self.self_consistency_error))
                break
            if line.rfind('--------------------------------------- Iteration') > -1:
                opt_iterations = int(line.split()[2].replace('(', ''))
            if line.rfind('   IBRION ') > -1:
                ibrion = int(line.split()[2])
            if line.rfind('   NSW    ') > -1:
                nsw = int(line.split()[2])
            if line.rfind('EDIFF  ') > -1:
                ediff = float(line.split()[2])
            if line.rfind('NELM') > -1:
                nelm = int(line.split()[2].replace(';', ''))
            if line.rfind('total energy-change') > -1:
                # I saw this in an atomic oxygen calculation. it
                # breaks this code, so I am checking for it here.
                if 'MIXING' in line:
                    continue
                split = line.split(':')
                a = float(split[1].split('(')[0])
                b = split[1].split('(')[1][0:-2]
                # sometimes this line looks like (second number wrong format!):
                # energy-change (2. order) :-0.2141803E-08  ( 0.2737684-111)
                # we are checking still the first number so
                # let's "fix" the format for the second one
                if 'e' not in b.lower():
                    # replace last occurrence of - (assumed exponent) with -e
                    bsplit = b.split('-')
                    bsplit[-1] = 'e' + bsplit[-1]
                    b = '-'.join(bsplit).replace('-e', 'e-')
                b = float(b)
                if [abs(a), abs(b)] < [ediff, ediff]:
                    self.completed = True
                else:
                    self.completed = False
                    continue

        outcar.close()

        # Then if ibrion in [1,2,3] check whether ionic relaxation
        # condition been fulfilled

        if (ibrion in [1, 2, 3, 7]) and (nsw not in [0]) and (self.completed):
            if opt_iterations < nsw:
                self.completed = True
            else:
                self.completed = False

        #        if (ibrion == -1) and (nsw == 0):
        #            if nelm == 1:
        #                outcar = open('./OUTCAR', 'r')
        #                for line in outcar.readlines():
        #                    if line.rfind('General timing and accounting informations for this job:') > -1:
        #                        self.completed = True

        if self.completed:
            self.completed = False

            outcar = open('./OUTCAR', 'r')
            for line in outcar.readlines():
                if line.rfind('General timing and accounting informations for this job:') > -1:
                    self.completed = True

        logger.info("VASP calculation completed successfully?     " + str(self.completed))
        logger.info("VASP crashed out due to error in SCF cycles? " + str(self.self_consistency_error))

    def execute(self):
        self.setup()
        self.run()
        self.check_convergence()

        if self.completed and self.clean_after_success:
            self.tear_down()
