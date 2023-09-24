def convert_poscar_to_geometry_input(poscar=None):
    from ase.io.vasp import read_vasp
    from ase.io.aims import write_aims
    system=read_vasp(poscar)
    write_aims('geometry.in',system)


if __name__=="__main__":
    import argparse,os
    parser = argparse.ArgumentParser(description='cmd utils for FHI-aims IO',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--convert_poscar",action='store_true',help='Convert POSCAR to geometry.in')
    args = parser.parse_args()

    if args.convert_poscar:
        if os.path.exists('./CONTCAR'):
            file='./CONTCAR'
            print("Input CONTCAR found!")
        elif os.path.exists('./POSCAR'):
            file = './POSCAR'
            print("Input POSCAR found!")
        else:
            raise Exception("No input found")

        convert_poscar_to_geometry_input(poscar=file)