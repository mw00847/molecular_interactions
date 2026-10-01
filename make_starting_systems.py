"""
making the folders and the initial systems 

LIG.gro is water 
UNK_5B7eB7.gro is acetone

functions.

A. calculate the volume of one molecule of substance used in the system. 

1. make_mol_fraction_dirs(mol_fractions, template_folder, working_base)
makes a new folder for each of the mol fractions and adds in the template files 

2. running_gromacs(mol_fractions, working_base)
creates the systems for each of the mol fractions, and then runs gromacs commands 

3.make_top()
makes the top file for each system based on inputs and compare with .gro files

"""
#imports 

import shutil 
import os 
import subprocess 

"""
SETTINGS
"""

#initialise gromacs 

os.environ["PATH"] = "/usr/local/gromacs/bin:" + os.environ["PATH"]
os.environ["GMXLIB"] = "/usr/local/gromacs/share/gromacs/top"

#name the working directory base 

working_base=("/home/molecular_interactions")

template_folder=("/home/molecular_interactions/template")

#experimental mol fractions 
mol_fractions= [round(i * 0.1, 2) for i in range(0, 11)]

avo = 6.02214076e23

def calc_volume_of_one_molecule(RMM,p):
     num_mol=1/avo
     mass=num_mol*RMM
     

     #GROMACS works in nm

     vol=(mass/p)*1e21
     return vol

V_WATER=calc_volume_of_one_molecule(18.015,0.997)
V_ACETONE=calc_volume_of_one_molecule(58.08,0.784)


"""
FUNCTIONS 
"""

#making folders of the mol fractions for simulation and the top file 
def make_mol_fraction_dirs(mol_fractions, template_folder, working_base):
    
    """
    create a new directory for each mol fraction by copying the template folder.
    """
    for i in mol_fractions:
        new_dir = f"{working_base}/x_acetone_{i}"

        #os.makedirs and shutil.copytree effectively do the same thing in making a new folder. but shutil is copying over the files into the new folder

        #process of making the new dirs
        #os.makedirs(new_dir,exist_ok=True)

        #dont copy the contents of the main folder you are looping through as this is just going to keep building what is then in the new folders, instead make a template folder to copy from. the template folder isnt going to change

        #copy the contents of the template directory into each mol fraction folder

        shutil.copytree(template_folder, new_dir)

    """
    how to make the top file for each folder

    with open("make_text.txt", "a") as writing:
    for i in portion:
        _ = writing.write(f"{i}@")

    """

def build_system(mol_fractions, working_base, total=1000, scale=1.15):
    """insert acetone then water into a box sized from the target density"""
    for x in mol_fractions:
        folder = f"{working_base}/x_acetone_{x}"
        n_ac = round(total * x)
        n_w = total - n_ac

        # box edge from ideal-mixing volume, scaled up so insertion has room
        L = f"{(scale * (n_ac * V_ACETONE + n_w * V_WATER)) ** (1 / 3):.3f}"

        if n_ac > 0 and n_w > 0:
            # mixture: acetone into an empty box, then water around it
            subprocess.run(
                ["gmx", "insert-molecules", "-ci", "UNK_5B7EB7.gro", "-nmol", str(n_ac),
                 "-box", L, L, L, "-o", "acetone.gro"],
                cwd=folder, check=True)
            subprocess.run(
                ["gmx", "insert-molecules", "-f", "acetone.gro", "-ci", "LIG.gro",
                 "-nmol", str(n_w), "-o", "system.gro"],
                cwd=folder, check=True)

        elif n_ac > 0:
            # pure acetone (x = 1.0): acetone only, straight to system.gro
            subprocess.run(
                ["gmx", "insert-molecules", "-ci", "UNK_5B7EB7.gro", "-nmol", str(n_ac),
                 "-box", L, L, L, "-o", "system.gro"],
                cwd=folder, check=True)

        else:
            # pure water (x = 0.0): water only, into an empty box
            subprocess.run(
                ["gmx", "insert-molecules", "-ci", "LIG.gro", "-nmol", str(n_w),
                 "-box", L, L, L, "-o", "system.gro"],
                cwd=folder, check=True)


def running_gromacs(mol_fractions, working_base):
    """energy minimisation, then NVT, then NPT in each mol fraction folder"""
    for x in mol_fractions:
        folder = f"{working_base}/x_acetone_{x}"

        # energy minimisation, starting from the built system
        subprocess.run(
            ["gmx", "grompp", "-f", "em.mdp", "-c", "system.gro",
             "-p", "topol.top", "-o", "em.tpr"],
            cwd=folder, check=True)
        subprocess.run(["gmx", "mdrun", "-deffnm", "em"], cwd=folder, check=True)

        # NVT, starting from the minimised structure
        subprocess.run(
            ["gmx", "grompp", "-f", "nvt.mdp", "-c", "em.gro",
             "-p", "topol.top", "-o", "nvt.tpr"],
            cwd=folder, check=True)
        subprocess.run(["gmx", "mdrun", "-deffnm", "nvt"], cwd=folder, check=True)

        # NPT, continuing from NVT (needs the checkpoint for velocities)
        subprocess.run(
            ["gmx", "grompp", "-f", "npt.mdp", "-c", "nvt.gro", "-t", "nvt.cpt",
             "-p", "topol.top", "-o", "npt.tpr"],
            cwd=folder, check=True)
        subprocess.run(["gmx", "mdrun", "-deffnm", "npt"], cwd=folder, check=True)
    

        # production, continuing from NPT
        subprocess.run(
            ["gmx", "grompp", "-f", "production.mdp", "-c", "npt.gro", "-t", "npt.cpt",
             "-p", "topol.top", "-o", "production.tpr"],
            cwd=folder, check=True)
        subprocess.run(["gmx", "mdrun", "-deffnm", "production"], cwd=folder, check=True)


        # clean up periodic boundaries on the finished trajectory
        subprocess.run(
            ["gmx", "trjconv", "-s", "production.tpr", "-f", "production.xtc",
             "-o", "production_pbc.xtc", "-pbc", "mol", "-center"],
            input="System\nSystem\n", text=True, cwd=folder, check=True)


def make_top(mol_fractions, working_base, total=1000):
    """write topol.top for each system; molecule counts must match system.gro"""
    for x in mol_fractions:
        n_acetone = round(total * x)
        n_water = total - n_acetone
        folder = f"{working_base}/x_acetone_{x}"

        with open(f"{folder}/topol.top", "w") as f:
            f.write(f"; Topology file for x_acetone_{x}\n")
            f.write('#include "oplsaa.ff/forcefield.itp"\n')
            f.write('#include "UNK_5B7EB7.itp"\n')
            f.write('#include "oplsaa.ff/tip3p.itp"\n\n')
            f.write("[ system ]\n")
            f.write(f"acetone water x={x}\n\n")
            f.write("[ molecules ]\n")
            f.write("; Compound\t nmols\n")
            if n_acetone > 0:
                f.write(f"UNK\t {n_acetone}\n")
            if n_water > 0:
                f.write(f"SOL\t {n_water}\n")


if __name__ == "__main__":
    make_mol_fraction_dirs(mol_fractions, template_folder, working_base)
    build_system(mol_fractions, working_base)
    make_top(mol_fractions, working_base)
    running_gromacs(mol_fractions, working_base)

