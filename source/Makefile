CC = nvcc
SRC = $(wildcard *.cu)
OBJ = $(SRC:.cu=.o)
EXEC = main 
<<<<<<< HEAD
#-arch=sm_35
LIBFLAGS =  -pg -g -Xptxas=-v #-arch=sm_35
=======
LIBFLAGS = -g -pg -Xptxas=-v #-arch=sm_35
>>>>>>> 2eb5f78a09b1d73a244a7b322f76da512073db8a
CCFLAGS = -rdc=true #-arch=sm_35
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