from opentrons import protocol_api
from opentrons.protocol_api import ALL, PARTIAL_COLUMN, SINGLE
from opentrons.types import Point
import time
import sys
import math
import random
import subprocess


metadata = {
    'protocolName': 'Clean 384 well DSF plate(s)',
    'author': 'Shawn Laursen',
    'description': '''
    ''',
    'apiLevel': '2.28'}

def add_parameters(parameters: protocol_api.Parameters):
    parameters.add_int(
        variable_name="num_plates",
        display_name="Number of plates",
        description="Number of 384 well qPCR plates to be washed",
        default=1,
        minimum=1,
        maximum=8,)

def run(protocol):
    protocol.set_rail_lights(True)
    setup(protocol)
    take_out_rxn(protocol)
    water_wash(protocol)
    protocol.set_rail_lights(False)

def setup(protocol):
    # equiptment
    global tips300, p300m, plates, troughs, waste
    tips300 = protocol.load_labware('opentrons_96_tiprack_300ul', 11)
    p300m = protocol.load_instrument('p300_multi_gen2', 'left', tip_racks=[tips300])
    plates = []
    for i in range(protocol.params.num_plates):
        plate = protocol.load_labware('appliedbiosystemsmicroamp_384_wellplate_40ul', i+1)
        plates.append(plate)
    res1 = protocol.load_labware('nest_1_reservoir_195ml', 10)
    waste = res1.wells()[0].top()
    troughs = []
    for j in range(7,10):
        trough = protocol.load_labware('nest_12_reservoir_15ml', j)
        troughs.append(trough)

def take_out_rxn(protocol):
    p300m.pick_up_tip()
    for plate in plates:
        for row in [0,1]:
            p300m.consolidate(20, plate.rows()[row][0:24], waste, new_tip='never')
    p300m.return_tip()

def water_wash(protocol):
    for trough in troughs:
        p300m.pick_up_tip()
        for plate in plates:
            for row in [0,1]:
                p300m.distribute(30, trough.wells()[plates.index(plate)], plate.rows()[row][0:24], new_tip='never')
        for plate in plates:
            for row in [0,1]:
                for well in range(24):
                    p300m.mix(3, 30, plate.rows()[row][well])
                p300m.consolidate(30, plate.rows()[row][0:24], waste, new_tip='never')            
        p300m.return_tip()
