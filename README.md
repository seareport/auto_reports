This script assesses the skill of our global model during 
 * [Storm Babet](https://en.wikipedia.org/wiki/Storm_Babet) 
 * [Storm Ciarán](https://en.wikipedia.org/wiki/Storm_Ciarán) events 
 
This example uses `analysea` and `pyposeidon`. to reproduce these scripts: 

    git clone git clone git@github.com:tomsail/analysea.git 
    poetry install 
    poetry add 'git+https://github.com/ec-jrc/pyposeidon.git'

plus some others libraries to connect to the Azure services

    poetry add adlfs zarr fsspec azure-identity

to run the storms, run: 

    python auto_reports.py

if you want to assess the skill of our global model for the last 30 days (by default), run:

    python plots.py