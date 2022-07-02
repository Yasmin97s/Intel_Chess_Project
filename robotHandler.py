import time
from positions import *

# we use this function to setup the game board for the initial state
def parking(swift):
    for i in range(0, 24):
        if legal_blue_positions[i]:
            move_and_pump(swift, parking_position, coordinates[i])
    x, y, z = exit
    swift.set_position(x, y, z, speed=50000)

# move a piece from source to dest
def move_and_pump(swift, dest, source):
    x, y, z = dest
    swift.set_position(x, y, 30, speed=50000)
    time.sleep(0.5)
    swift.set_position(x, y, 1.2, speed=10000)
    swift.set_speed_factor(0.5)
    swift.set_mode(mode=0)
    swift.set_wrist(90)
    # swift.set_servo_attach()
    # swift.set_servo_detach()
    # swift.set_buzzer(frequency=1000, duration=2)
    swift.set_pump(on=True)
    swift.set_mode(mode=0)
    swift.set_servo_attach(wait=True, timeout=None)
    time.sleep(0.5)
    swift.set_position(x, y, 30, speed=10000)
    # swift.set_position(x=200, y=0, z=150, speed=50000, wait=True, timeout=None)
    x, y, z = source
    swift.set_position(x, y, 30, speed=50000)
    time.sleep(0.5)
    swift.set_position(x, y, 2.91, speed=10000, wait=True)
    time.sleep(1)
    swift.set_pump(on=False, timeout=1)
    swift.set_servo_attach(wait=True, timeout=None)
    swift.set_position(x, y, 30, speed=10000, wait=True)
    time.sleep(0.5)

# robot move the piece that it killed outside the game board
def move_to_death(swift, dest, source=cemetry):
    x, y, z = dest
    swift.set_position(x, y, 30, speed=50000)
    time.sleep(0.5)
    swift.set_position(x, y, 1.2, speed=10000)
    swift.set_speed_factor(0.5)
    swift.set_mode(mode=0)
    swift.set_wrist(90)
    # swift.set_servo_attach()
    # swift.set_servo_detach()
    # swift.set_buzzer(frequency=1000, duration=2)
    swift.set_pump(on=True)
    swift.set_mode(mode=0)
    swift.set_servo_attach(wait=True, timeout=None)
    time.sleep(0.5)
    swift.set_position(x, y, 30, speed=10000)
    # swift.set_position(x=200, y=0, z=150, speed=50000, wait=True, timeout=None)
    x, y, z = source
    time.sleep(0.5)
    swift.set_position(x, y, 40, speed=10000, wait=True)
    time.sleep(1)
    swift.set_pump(on=False, timeout=1)
    swift.set_servo_attach(wait=True, timeout=None)
    swift.set_position(x, y, 30, speed=10000, wait=True)
    time.sleep(0.5)