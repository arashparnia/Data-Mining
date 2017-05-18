from oct2py import octave
octave.addpath('/')

def nDCG(predictions,actuals,queryIds):
    n = octave.nDCG( predictions, actuals, queryIds )
    return n
