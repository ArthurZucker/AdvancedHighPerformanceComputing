# Batch merge and merge sort using CUDA programming
GPU project on CUDA, implemented by  `Arthur Zucker` & `Clément Apavou` (MAIN5). This was the final project of our 5th year's Advanced High Performance Computing class.  
The different questions we answered can be found under the *Projet.pdf* file. 
We used different types of memories to compare our implementation, and tested most of our code on our unviversity's GPU.
Our results can be found in the *result* folder. 
# Structure
```
├── results                 # Result files
│   ├── graph_on_beamer     # graph on the beamer presentation
│   ├── *.csv               # results written using the `main.cu`
│   ├── *.svg               # figures
│   ├── plot_data.py        # Used to get the svg files
├── source                  # Source files
│   ├── Doxyfile            # for doxygen documentation
│   ├── html                # hmtl doxygen documentation, see `index.html` for the link
│   ├── latex               # documentation
│   ├── Makefile            # Makefile called using `make`
│   ├── *.cu                # cuda files , merge for the part 1 and batch_merge for the part 2, utils (several function that we used)
│   ├── *.h                 # headers
├── *.pdf                   # beamer latex for the presentation of the project
└── README.md
```
# Utilisation
On line 35 of the main, you have to choose from {1,2,3,4,5} depending on the question that you want to launch.
