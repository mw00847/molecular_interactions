"""
making the folders and the initial systems 

LIG.gro is water 
UNK_5B7EB7.gro is acetone

"""
#imports 

import shutil 
import os 
import subprocess 

#initialise gromacs 

os.environ["PATH"] = "/usr/local/gromacs/bin:" + os.environ["PATH"]
os.environ["GMXLIB"] = "/usr/local/gromacs/share/gromacs/top"

#name the working directory base 

working_base=("MD")

template_folder=("MD/template")

#experimental mol fractions 
mol_fractions= [round(i * 0.1, 2) for i in range(1, 11)]

"""
function to calculate the volume of one molecule used in the system acetone/water
function takes RMM and density (p) and avogadros constant (avo)

p=mv , mole=mass/RMM , molecules=moles*avo

RMM water = 18.015
RMM acetone = 58.08

p water = 0.997
p acetone = 0.784

"""

avo=6.02214076e23

def calc_volume_of_one_molecule(RMM,p):
    
    num_mol=1/avo

    mass=num_mol*RMM

    #gromacs takes nm

    vol=(mass/p)*1e21
    return vol

#using the function with the RMM and p inputs for the two molecules 
V_WATER=calc_volume_of_one_molecule(18.015,0.997)
V_ACETONE=calc_volume_of_one_molecule(58.08,0.784)


#making folders of the mol fractions for simulation and the top file 

for i in mol_fractions:
    new_dir=f"MD/x_acetone_{i}"

    """
    os.makedirs and shutil.copytree effectively do the same thing in making a new folder. but shutil is copying over the files into the new folder"""

    #process of making the new dirs 
    #os.makedirs(new_dir,exist_ok=True)

    """
    dont copy the contents of the main folder you are looping through as this is just going to keep building what is then in the new folders, instead make a template folder to copy from. the template folder isnt going to change"""

    #copy the contents of the template directory into each mol fraction folder
    
    shutil.copytree(template_folder,new_dir)

    """
    how to make the top file for each folder

    with open("make_text.txt", "a") as writing:
    for i in portion:
        _ = writing.write(f"{i}@")

    """


"""
making a new loop or a function to run the create systems

"""

for j in mol_fractions:

    #loop through the mol_fraction list

    folder=f"MD/x_acetone_{j}"

    """
    run gromacs commands 
    
    make a box with acetone molecule and put it in a box of 4x4x4
    make the right size boxes and number of molecules 
    """

    subprocess.run(
            ["gmx", "editconf", "-f", "UNK_5B7EB7.gro", "-o", "box.gro", "-box", "4", "4", "4"]
            ,cwd=folder
            )
