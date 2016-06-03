CC = gcc
CFLAGS = -g -Wall -pedantic -Wstrict-prototypes -lm -Ofast

#command: dependencies
#	 $(CC) $(CFLAGS) programs.c

mhd: numerics_cpu.o mhd.o
	$(CC) $(CFLAGS) -o mhd numerics_cpu.o mhd.o

tests: numerics_cpu.o tests.o
	$(CC) $(CFLAGS) -o tests numerics_cpu.o tests.o

%.o: %.cpp
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o tests mhd aaa
	
.PHONY: all clean

