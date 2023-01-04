from Reader import JoyStickReader
from BrainMain import BrainComputerInterface

def main():
    # input size 16 * 64
    # output size 12
    # layer 125
    bm = BrainComputerInterface(32*60,12,125,12)
    joystickReader = JoyStickReader(bm)
    joystickReader.readJoyStick()

if __name__ == '__main__':
    main()


