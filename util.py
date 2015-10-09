class Params:
     def __init__(self, dictionary):
         for k, v in dictionary.items():
             setattr(self, k, v)

#TODO: Make a small util program that create the dataset structure
