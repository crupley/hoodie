
from code.makedbs import make_all_db
from code.featurize import featurizer

"""
Main execution file for hoodie project
Intermediate results are saved after each step
"""

if __name__ == '__main__':
    db = raw_input('Create psql databases? y: Proceed, other: Skip > ')
    if db == 'y':
        execfile('code/makedbs.py')

    ft = raw_input('Create features? y: Proceed, other: Skip > ')
    if ft == 'y':
        execfile('code/featurizer.py')

    gr = raw_input(('Create graphs and cutlists?\n'
                    'Note: calculation can take approx. 1-5 hours per graph, '
                    'total of 130 graphs\ny: Proceed, other: Skip > '))
    if gr == 'y':
        execfile('code/graphreduce.py')

    mg = raw_input('Create geojson file? y: Proceed, other: Skip > ')
    if mg == 'y':
        execfile('code/clusterize.py')
