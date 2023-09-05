
# MapMEQ

MapMEQ was developed by Jeffrey R Bailey for the 2023 SPE Geothermal Datathon, as part
of the ThermoSeekers project team.

The application is distributed in 3 directories.
..\ the root directory holds the license and readme files.  The app is offered on Github under MIT License terms.
..\code\ contains the Python source code files for MapMEQ.  This code was developed under the Anaconda distribution of Python 3.9.12 (Apr 4, 2022).
..\runs\ is the location for run files, including the FORGE dataset.  Results and charts are stored here.

The main code is MapMEQ.py.  When executed, it calls RunMEQ.py which is the main source code for the various procedures.
The code will make a date-stamped directory such as ..\runs\MEQ_2023-09-05\ in which all the output and temporary files will be stored.

## License

See the [LICENSE](LICENSE.md) file for license rights and limitations (MIT).
