# ======================PRINTING UTILITIES============================
'''
Utilities to print formatted text in console. IE, text that has a certain color, section headers and more.
'''
class Color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def print_color(text):
    print(Color.PURPLE + text + Color.END)

def print_action(text):
    print(Color.PURPLE + text + Color.END)


def print_error(text):
    print(Color.RED + text + Color.END)

def print_section(description):
    print('')
    print('')
    print('')
    print(Color.BOLD + '========== ' + str(description) + ' ==========' + Color.END)


def print_test(loss):
    #print('Epoch {}, minibatch {}/{}'.format(epoch, idx, minibatches))
    print('---- Test error %f MSE' % (loss))

def print_training(loss):
    print('---- Training error %f MSE' % (loss))
    print('')


def print_valid(epoch, idx, minibatches, loss):
    print('')
    print(Color.CYAN + 'Epoch {}, minibatch {}/{}'.format(epoch, idx, minibatches) + Color.END)
    print('---- Validation error %f MSE' % (loss))