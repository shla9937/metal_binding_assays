from opentrons import protocol_api
from opentrons.protocol_api import ALL, PARTIAL_COLUMN, SINGLE
from opentrons.types import Point
import time
import sys
import math
import random
import subprocess


metadata = {
    'protocolName': 'DSF - 384 well, 6 metal quadruplicate',
    'author': 'Shawn Laursen',
    'description': '''
    Adds buff + spyro + protein
    Titrates 8 metals in 12 point dilution series. 
    For 1:1 dilution - (100µM, 50.0µM, 25.0µM, 12.5µM, 6.25µM, 3.13µM, 1.56µM, 781nM, 391nM, 195nM, 97.7nM, 48.8nM)
    * can do 6 metals, EDTA and then a buffer (aka Apo protein)
    Stocks:
    -   Metal: 5x (500µM) -> 100µM final (50µL into wells)
    -   EDTA: 5x (500µM) -> 100µM final (50µL into well G1)
    -   Buff 1x  (50µL into well H1 and 10mL in well 1 of trough)
    -   Protein + SYPRO + ROX: 5x (25µM, 50x, 250nM) -> 5µM, 10x, 50nM final (2mL total -> 250µL into wells of col 12 of 96 well plate)

    Buff should be 100mM buff, 150mM NaCl, pH 6 or lower of a Good's buff''',
    'apiLevel': '2.26'}

def run(protocol):
    protocol.set_rail_lights(True)
    setup(protocol)
    add_protein_and_sypro(protocol) 
    add_metal_and_titrate(protocol)
    protocol.set_rail_lights(False)

def setup(protocol):
    # equiptment
    global tips20, metals, plate, trough, p20m
    tips20 = protocol.load_labware('opentrons_96_tiprack_20ul', 2)
    metals = protocol.load_labware('greiner_96_wellplate_300ul', 4)
    plate = protocol.load_labware('corning_384_wellplate_112ul_flat', 5) 
    trough = protocol.load_labware('nest_12_reservoir_15ml', 6)
    p20m = protocol.load_instrument('p20_multi_gen2', 'right', tip_racks=[tips20])
         
    # reagents     
    global buff, protein_and_sypro
    buff = trough.wells()[0]
    protein_and_sypro = metals.rows()[0][11]

    # rows
    global rxn_vol, start_vol, dilution_factor
    rxn_vol = 20   
    dilution_factor = 1 # i.e. 1:2, not 1 in 2
    start_vol = rxn_vol + (rxn_vol/dilution_factor)

def add_protein_and_sypro(protocol):
    rows = [0,1,0,1]
    cols = [0,0,12,12]
    
    p20m.pick_up_tip()
    for row, col in zip(rows, cols):
        p20m.transfer(start_vol*(3/5), buff, plate.rows()[row][col], new_tip='never')
        p20m.transfer(rxn_vol*(4/5), buff, plate.rows()[row][col+1:col+12], new_tip='never')
    for row, col in zip(rows, cols):
        p20m.transfer(start_vol*(1/5), protein_and_sypro, plate.rows()[row][col], new_tip='never')
        p20m.transfer(rxn_vol*(1/5), protein_and_sypro, plate.rows()[row][col+1:col+12], new_tip='never')
    p20m.return_tip()

def add_metal_and_titrate(protocol):
    rows = [0,1,0,1]
    cols = [0,0,12,12]

    for row, col in zip(rows, cols):
        p20m.pick_up_tip()
        p20m.transfer(start_vol*(1/5), metals.rows()[0][0], plate.rows()[row][col], new_tip='never', 
                mix_before=(3,rxn_vol))
        p20m.transfer(rxn_vol/dilution_factor, plate.rows()[row][col+0:col+11], plate.rows()[row][col+1:col+12], 
                    mix_before=(3,rxn_vol), new_tip='never')    
        p20m.mix(3,rxn_vol, plate.rows()[row][col+11])
        p20m.aspirate(rxn_vol/dilution_factor, plate.rows()[row][col+11])
        p20m.return_tip()
