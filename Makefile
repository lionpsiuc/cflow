CC = gcc
CFLAGS = -funroll-loops -march=native -O3 -std=c2x -Wall -Wextra

TARGET = assignment2

SRCS = assignment2.c average.c iteration.c utils.c
HDRS = average.h iteration.h utils.h
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ -lm

assignment2.o: assignment2.c $(HDRS)
	$(CC) $(CFLAGS) -c $< -o $@

average.o: average.c average.h
	$(CC) $(CFLAGS) -c $< -o $@

iteration.o: iteration.c iteration.h
	$(CC) $(CFLAGS) -c $< -o $@

utils.o: utils.c utils.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean
