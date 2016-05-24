import sys, os

sys.path.append(os.path.abspath("./"))
import printing

def get_command(cmd, default=None):
    '''
    Method for finding a console command and returning its value. A default string can be supplied, which is returned
    if command is not found.
    :param cmd: Command that should be found in system arguments.
    :param default: Returned value if command not found.
    :return:
    '''
    is_cmd = False
    value = default
    if cmd in sys.argv:
        is_cmd = True
        idx = sys.argv.index(cmd)
        if len(sys.argv) > idx+1:
            v = sys.argv[idx+1]
            if v[0] != '-':
                value = v
        printing.print_action("COMMAND: {}, VALUE: {}".format(cmd, value))
    return is_cmd, value
