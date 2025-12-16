from pymatgen.core.structure import Structure
from core.data.element import pbe_pp_choices
import os


class Potcar:
    """
    A lightweighted writer for preparing VASP POTCAR files using PBE (GW) pseudopotentials.
    Re-implementing the futuremate dao implementation but using pymatgen Structure objects, so it
    can be used without the functionality of futuremate.

    The key point is to ensure that the code picked up the correct pseudopotential files as specified by
    the default Materials Project PBE pseudopotential choices, and from the local directories in which
    the PAW pseudopotential files are stored.

    """

    def __init__(self, structure: Structure, vasp_pp_directory: str = None):
        """
        :param structure: The pymatgen Structure object representing the atomic structure
        :type structure: Structure
        :param vasp_pp_directory: The directory where VASP pseudopotential files are stored. Bt default, it will try
        to pick up this from settings.py.
        :type vasp_pp_directory: str
        """
        self.structure = structure
        if vasp_pp_directory is not None:
            self.vasp_pp_directory = vasp_pp_directory
        else:
            try:
                from settings import vasp_pp_directory

                self.vasp_pp_directory = vasp_pp_directory
            except ImportError:
                raise ImportError(
                    "Please set the vasp pseudopotential directory in settings.py or provide it as an argument."
                )

    def write(self, use_GW: bool = False) -> None:
        """
        Write a combined POTCAR file by concatenating individual POTCAR files for each element in the structure.
        This method extracts unique element symbols from the structure, retrieves the corresponding POTCAR files
        from the VASP pseudopotential directory, and concatenates them into a single POTCAR file. It supports
        both standard PBE and GW pseudopotentials.
        Args:
            use_GW (bool, optional): If True, attempts to use GW-specific pseudopotentials if available.
                                     Falls back to standard PBE potentials if GW variants don't exist.
                                     Defaults to False.
        Returns:
            None
        Notes:
            - Creates a "POTCAR" file in the current working directory
            - Requires self.vasp_pp_directory to be set with the path to pseudopotential files
            - Requires pbe_pp_choices dictionary to map element symbols to pseudopotential names
        Raises:
            FileNotFoundError: If a required POTCAR file cannot be found
            IOError: If unable to read input POTCAR files or write to output POTCAR file
        """

        ordered_symbols = []
        for site in self.structure.sites:
            symbol = site.specie.symbol
            sym = (
                site.specie.symbol
                if hasattr(site, "specie")
                else str(site.species_string)
            )

            if sym not in ordered_symbols:
                ordered_symbols.append(sym)

        if not use_GW:
            potcars = [
                self.vasp_pp_directory + "/" + pbe_pp_choices[e] + "/POTCAR"
                for e in ordered_symbols
            ]
        else:
            potcars = []
            import os

            for e in ordered_symbols:
                if os.path.exists(
                    self.vasp_pp_directory + "/" + pbe_pp_choices[e] + "_GW/"
                ):
                    potcars.append(
                        self.vasp_pp_directory + "/" + pbe_pp_choices[e] + "_GW/POTCAR"
                    )
                else:
                    potcars.append(
                        self.vasp_pp_directory + "/" + pbe_pp_choices[e] + "/POTCAR"
                    )

        with open("POTCAR", "w") as outfile:
            for fn in potcars:
                with open(fn) as infile:
                    for line in infile:
                        if ("Zr" in fn) and ("VRHFIN" in line):
                            line = "   VRHFIN =Zr: 4s4p5s4d\n"
                        outfile.write(line)
