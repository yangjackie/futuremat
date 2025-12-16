import logging

logger = logging.getLogger("futuremat.core.calculators")

import os


class Calculator:

    def __init__(self):
        self.gpu_run = False
        self.kpoint_mode = None

    @property
    def structure(self):
        """
        Get the structure for the calculation.
        """
        if hasattr(self, "_structure") is False:
            raise Exception("Structure has not been set for the calculator!")
        return self._structure

    @structure.setter
    def structure(self, val):
        """
        Set the structure for the calculation.
        """
        print("Setting structure for the calculator.", val)
        self._structure = val


class VaspBase(Calculator):

    def set_incar_params(self, **kwargs):
        """
        From the provided keyword arguments, filter out those that belongs to the VASP INCAR keys.
        """
        from core.data.vasp_infos import all_incar_keys

        self.incar_params = {}
        for key in kwargs.keys():
            if key.lower() in all_incar_keys:
                self.incar_params[key.lower()] = kwargs[key]
        self.incar_params = dict(sorted(self.incar_params.items()))

    @property
    def executable(self):
        """
        Set the VASP executable based on some user settings.
        """
        # TODO: expand this to more options as needed and make it generalised
        if not self.gpu_run:
            if self.kpoint_mode == "gamma":
                self._executable = "vasp_gam"
            else:
                self._executable = "vasp_std"
        else:
            if self.kpoint_mode == "gamma":
                self._executable = "vasp_gam-gpu"
            else:
                self._executable = "vasp_std-gpu"

        logger.info(
            "VASP calculation to be executed with the following binary: "
            + str(self._executable)
        )

        if self._executable == None:
            raise Exception("Problem setting the VASP executable!")

        return self._executable

    @property
    def command(self):
        """
        Set up the command that execute the VASP calculation. This is implemented specifically for different
        HPCs and how VASP is actually compiled on that machine.

        Returns
        -------
        str
            The command string to execute the VASP calculation.

        """

        hpc_name = os.environ.get("PBS_O_HOST")
        if hpc_name is None:
            return "NONE"

        if "katana" in hpc_name:
            if self.gpu_run:
                logger.info(
                    "Choose to run with GPU, automatically set mpirun -np $NGPUS"
                )
                self._command = "mpirun -np $NGPUS " + self.executable
            else:
                logger.info(
                    "Choose to run with CPU, automatically set mpirun -np $NCPUS"
                )
                self._command = "mpirun -np $NCPUS " + self.executable

        if self._command is None:
            raise Exception("Problem setting the VASP execution command!")
            pass
        return self._command

    @command.setter
    def command(self, val):
        """
        Set the VASP execution command.

        Parameters
        ----------
        val : str
            The command string to use for executing VASP calculations.
        """

        self._command = val

    def run(self):
        logger.info("Start executing VASP")
        exitcode = os.system("%s > %s" % (self.command, "vasp.log"))
        if exitcode != 0:
            raise RuntimeError("Vasp exited with exit code: %d.  " % exitcode)
