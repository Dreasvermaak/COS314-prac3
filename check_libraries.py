try:
    import numpy
    import pandas
    from sklearn.neural_network import MLPClassifier
    print('All required libraries are installed')
    exit(0)
except ImportError as e:
    print('Missing library:', e)
    exit(1)
