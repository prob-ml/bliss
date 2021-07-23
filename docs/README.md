To rebuild the Sphinx documentation:

1. `cd docsrc`
2. `make clean` to clean the html files that are already existed under `/build`
3. `make github` to create html files and move to `docs`

The metadata for each module lives in ```docsrc/api/```.
A particular file is added to the documentation by editing the entries of the ```.rst``` files.
