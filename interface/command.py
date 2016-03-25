import sys, os

sys.path.append(os.path.abspath("./"))
import printing

def get_command(cmd, default=None):
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
