CC = nvcc
SRC = $(wildcard *.cu)
OBJ = $(SRC:.cu=.o)
EXEC = main 
LIBFLAGS = -g -pg -Xptxas=-v #-arch=sm_35
CCFLAGS =  -rdc=true #-arch=sm_35
all : $(EXEC)

$(EXEC) : $(OBJ)
	$(CC) $(LIBFLAGS) $^ -o $@  
    
%.o : %.cu
	$(CC) $(CCFLAGS) -c $< -o $@ 
    
.depend :
	@gcc -MM $(SRC) > .depend
-include .depend

clean : 
	rm -rf $(EXEC) *.o