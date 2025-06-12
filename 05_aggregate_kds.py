#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate Kd values from all DSF experiments into a master table."
    )
    parser.add_argument('-d', '--directory', type=str, required=True, 
                       help="Root directory containing all DSF experiment folders")
    args = parser.parse_args()

    # Find all tm_values.csv files
    tm_files = []
    for root, dirs, files in os.walk(args.directory):
        for file in files:
            if file.endswith('_tm_values.csv'):
                tm_files.append(os.path.join(root, file))

    # Initialize empty dictionary to store protein:metal:kd mappings
    kd_dict = {}
    metal_set = set()

    # Process each file
    for file in tm_files:
        # Extract protein ID from directory path
        protein_id = os.path.basename(os.path.dirname(file)).split('_')[0]
        
        # Read the CSV file
        df = pd.read_csv(file, index_col=0)
        
        # Get Kd values
        kd_row = df.loc['Kd']
        
        # Store values for protein, excluding control columns
        kd_dict[protein_id] = {}
        for metal in kd_row.index:
            if metal not in ['WT', 'EDTA', 'Blank'] and ('HCl' not in metal):
                value = f"{float(kd_row[metal]):.10f}"
                kd_dict[protein_id][metal] = value
                metal_set.add(metal)

    # Sort proteins by ID number
    def sort_key(x):
        digits = ''.join(filter(str.isdigit, x))
        return int(digits) if digits else float('inf')  # Return infinity if no digits
    
    sorted_proteins = sorted(kd_dict.keys(), key=sort_key)
    
    # Define periodic table structure for atomic number lookup
    period_elements = {
        1: ['H', 'He'],
        2: ['Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne'],
        3: ['Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar'],
        4: ['K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'],
        5: ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
            'In', 'Sn', 'Sb', 'Te', 'I', 'Xe'],
        6: ['Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
            'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
            'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'],
        7: ['Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
            'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
            'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
    }

    # Create atomic numbers dictionary
    atomic_numbers = {}
    z = 1
    for period in sorted(period_elements.keys()):
        for element in period_elements[period]:
            atomic_numbers[element] = z
            z += 1

    def get_metal_symbol(col):
        return ''.join(c for c in col if c.isalpha())

    def get_oxidation_state(col):
        superscript_map = {'¹': 1, '²': 2, '³': 3, '⁴': 4, '⁵': 5, '⁶': 6, '⁷': 7}
        for char in col:
            if char in superscript_map:
                return superscript_map[char]
        return 1 if '⁺' in col else 0

    def get_sort_key(metal):
        oxidation_state = get_oxidation_state(metal)
        symbol = get_metal_symbol(metal)
        atomic_num = atomic_numbers.get(symbol, float('inf'))
        return (atomic_num, oxidation_state)

    # Sort metals by atomic number and oxidation state
    sorted_metals = sorted(metal_set, key=get_sort_key)

    # Create DataFrame
    master_df = pd.DataFrame(index=sorted_proteins, columns=sorted_metals)
    
    # Fill DataFrame with Kd values
    for protein in sorted_proteins:
        for metal in sorted_metals:
            master_df.loc[protein, metal] = kd_dict[protein].get(metal, '')

    # Save to CSV with blank cells instead of NaN
    output_path = os.path.join(args.directory, 'experimental_kds.csv')
    master_df.to_csv(output_path, na_rep='', float_format='%.10f')

if __name__ == '__main__':
    main()

