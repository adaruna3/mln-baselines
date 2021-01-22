from argparse import ArgumentParser
from sys import stdin
from select import select

class ArgParse:
    def __init__(self, description):
        self.parser = ArgumentParser(description=description)
        self.parser.add_argument('input_file', type=str, help='Dataset filename (.txt)')
        self.parser.add_argument('output_file', type=str, help='Database filename (.db)')
        self.parser.add_argument('-m', dest='mode', type=str, nargs='?',
                                 default="train", help='Database type: train, test')

    def parse(self):
        """ prints the current command-line options set, waiting 10 s before continuing """
        parsed_args = self.parser.parse_args()
        if parsed_args.sess_mode:
            flag = "non-test"
        else:
            flag = "test"
        print("The current " + flag + "ing parameters are: \n" + str(parsed_args))

        if not self.confirm_args():
            exit()

        return parsed_args

    def confirm_args(self):
        """ captures input from user for 10 s, if no input provided returns """
        print('Continue? (Y/n) waiting 10s ...')
        i, o, e = select([stdin], [], [], 10.0)
        if i:  # read input
            cont = stdin.readline().strip()
            if cont == 'Y' or cont == 'y' or cont == '':
                return True
            else:
                return False
        else:  # no input, start training
            return True