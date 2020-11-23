CC = nvcc
SRC = $(wildcard *.cu)
OBJ = $(SRC:.cu=.o)
EXEC = main
LIBFLAGS = -g 
CCFLAGS = -rdc=true
all : $(EXEC)

$(EXEC) : $(OBJ)
	$(CC) $(LIBFLAGS) $^ -o $@  
    
%.o : %.cu
	$(CC) $(CCFLAGS) -o $@ -c $<
    
.depend :
	@gcc -MM $(SRC) > .depend
-include .depend

clean : 
	rm -rf $(EXEC) *.o