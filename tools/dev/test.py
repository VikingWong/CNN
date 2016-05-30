__author__ = 'olav'

with open('/home/olav/Desktop/sanitized.txt', 'a') as w:
    with open('/home/olav/Desktop/thesis.txt') as fp:
        for line in fp:
            if '' not in line:
                w.write(line)
