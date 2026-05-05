from opentrons import protocol_api
from opentrons.protocol_api import ALL, PARTIAL_COLUMN, SINGLE
from opentrons.types import Point
import time
import sys
import math
import random
import subprocess


metadata = {
    'protocolName': 'DSF - 3 x 384 well, 30 metals',
    'author': 'Shawn Laursen',
    'description': '''
    Adds buff + spyro + protein
    Titrates 29 metals in 12 point dilution series. 
    For 1:2 dultion - (1mM, 333µM, 111µM, 37µM, 12.3µM, 4.1µM, 1.37µM, 457nM, 152nM, 51nM, 17nM, 6nM)
    For 1:1 dilution - (100µM, 50.0µM, 25.0µM, 12.5µM, 6.25µM, 3.13µM, 1.56µM, 781nM, 391nM, 195nM, 97.7nM, 48.8nM)

    Stocks:
    -   metal: 5x (5mM/500µM) -> 1mM/100µM final (into 15mL Falcons)
    -   EDTA: 5x (500mM/500µM) -> 100mM/100µM final (into last 15mL Falcon)
    -   Apo: buff (need ~10mL in wells 1-4  of trough)
    -   protein + sypro + rox: 5x (25µM, 50x, 250nM) -> 5µM, 10x, 50nM final (6mL total -> 250µL into last 3 columns of 96well)

    Buff should be ~100mM buff, 150mM NaCl (10mL in trough)''',
    'apiLevel': '2.26'}

def run(protocol):
    protocol.set_rail_lights(True)
    setup(protocol)
    dilute_metals(protocol)
    for iteration in range(3):
        add_protein_and_sypro(protocol, iteration) 
        add_metal_and_titrate(protocol, iteration)
        message(protocol, iteration)
    protocol.set_rail_lights(False)

def setup(protocol):
    # equiptment
    global tips20, tips20_2, tips300, metals, plates, trough, p20m, p300s, tubes1, tubes2
    tips20 = protocol.load_labware('opentrons_96_tiprack_20ul', 11)
    tips20_2 = protocol.load_labware('opentrons_96_tiprack_20ul', 10)
    tips300 = protocol.load_labware('opentrons_96_tiprack_300ul', 5)
    metals = protocol.load_labware('greiner_96_wellplate_300ul', 4)
    plates = []
    for i in [1,2,3]:
        plate = protocol.load_labware('appliedbiosystemsmicroamp_384_wellplate_40ul', i) 
        plates.append(plate)
    trough = protocol.load_labware('nest_12_reservoir_15ml', 6)
    p20m = protocol.load_instrument('p20_multi_gen2', 'right', tip_racks=[tips20, tips20_2])
    p300s = protocol.load_instrument('p300_single_gen2', 'left', tip_racks=[tips300])
    tubes1 = protocol.load_labware('opentrons_15_tuberack_falcon_15ml_conical', 7)
    tubes2 = protocol.load_labware('opentrons_15_tuberack_falcon_15ml_conical', 8)
         
    # reagents     
    global buff, buffs, proteins
    buff = trough.wells()[0]
    buffs = []
    for i in [1,2,3]:
        buff_i = trough.wells()[i]
        buffs.append(buff_i)
    proteins = []
    for i in [9,10,11]:
        protein = metals.rows()[0][i]
        proteins.append(protein)

    # rows
    global rxn_vol, start_vol, dilution_factor
    rxn_vol = 20   
    dilution_factor = 1 # i.e. 1:2, not 1 in 2
    start_vol = rxn_vol + (rxn_vol/dilution_factor)

def dilute_metals(protocol):
    # add buff to wells
    p300s.pick_up_tip()
    p300s.transfer(190, buff, metals.wells()[0:31], new_tip='never')
    p300s.transfer(200, buff, metals.wells()[31], new_tip='never')
    p300s.return_tip()

    # add metal to buffs
    well = 0
    for rack in [tubes1, tubes2]:
        for tube in range(15):
            p300s.pick_up_tip()
            p300s.transfer(10, rack.wells()[tube], metals.wells()[well], new_tip='never', mix_after=(3,50))         
            p300s.return_tip()   
            well += 1
            
    # add extra EDTA well
    p300s.pick_up_tip()
    p300s.transfer(10, tubes2.wells()[14], metals.wells()[30], new_tip='never', mix_after=(3,50))         
    p300s.return_tip()   

def add_protein_and_sypro(protocol, iteration):
    rows = [0,1,0,1]
    cols = [0,0,12,12]
    
    p20m.pick_up_tip()
    for row, col in zip(rows, cols):
        p20m.transfer(start_vol*(3/5), buffs[iteration], plates[iteration].rows()[row][col], new_tip='never')
        p20m.transfer(rxn_vol*(4/5), buffs[iteration], plates[iteration].rows()[row][col+1:col+12], new_tip='never')
    for row, col in zip(rows, cols):
        p20m.transfer(start_vol*(1/5), proteins[iteration], plates[iteration].rows()[row][col], new_tip='never')
        p20m.transfer(rxn_vol*(1/5), proteins[iteration], plates[iteration].rows()[row][col+1:col+12], new_tip='never')
    p20m.return_tip()

def add_metal_and_titrate(protocol, iteration):
    rows = [0,1,0,1]
    cols = [0,0,12,12]
    metal_col = [0,1,2,3]

    for row, col, metal in zip(rows, cols, metal_col):
        p20m.pick_up_tip()
        p20m.transfer(start_vol*(1/5), metals.rows()[0][metal], plates[iteration].rows()[row][col], new_tip='never', 
                mix_before=(3,rxn_vol))
        p20m.transfer(rxn_vol/dilution_factor, plates[iteration].rows()[row][col+0:col+11], plates[iteration].rows()[row][col+1:col+12], 
                    mix_before=(3,rxn_vol), new_tip='never')    
        p20m.mix(3,rxn_vol, plates[iteration].rows()[row][col+11])
        p20m.aspirate(rxn_vol/dilution_factor, plates[iteration].rows()[row][col+11])
        p20m.return_tip()

def message(protocol, iteration):
    if iteration != 2:
        protocol.pause("Read plate in qPCR. Next plate will start after this pause.")
    else:
        protocol.pause("Read plate in qPCR.")