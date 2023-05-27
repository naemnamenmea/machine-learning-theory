import os
from builtins import print

import plotly
import plotly.graph_objs as go
import urllib.request as urllib2

"""
export HTTP_PROXY="http://user:pass@10.10.1.10:3128/"

def parseAndSetEnv(value):
    print 'export MYVAR="%s"' % value
"""
os.environ['HTTP_PROXY'] = 'http://207.144.111.230:8080'
os.environ['HTTPS_PROXY'] = 'https://176.56.71.211:60751'

plotly.tools.set_credentials_file(username='naemnamenmea', api_key='NOkbqklPj3PNQpjnH2q0')
#plotly.plotly.sign_in(username ='naemnamenmea', api_key ='NOkbqklPj3PNQpjnH2q0')

trace0 = go.Scatter(
    x=[1, 2, 3, 4],
    y=[10, 15, 13, 17]
)
trace1 = go.Scatter(
    x=[1, 2, 3, 4],
    y=[16, 5, 11, 9]
)
data = [trace0, trace1]

plotly.plotly.plot(data, filename = 'basic-line')
