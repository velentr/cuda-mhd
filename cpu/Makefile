CC = gcc
CFLAGS = -g -Wall -ansi -pedantic -Wstrict-prototypes -lm

#command: dependencies
#	 $(CC) $(CFLAGS) programs.c

tests: numerics_cpu.o tests.o
	$(CC) $(CFLAGS) -o tests numerics_cpu.o tests.o
%.o: %.cpp
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o tests
	
.PHONY: all clean

