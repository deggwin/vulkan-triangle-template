ifeq ($(OS), Windows_NT)
 LIBS = -lvulkan-1 -lglfw3 -lopengl32 -lgdi32
 WIN = true
else
 UNAME_S = $(shell uname -s)
 ifeq ($(UNAME_S), Linux)
 	LIBS = -lvulkan -lglfw -lGL -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi
 	WIN = false
 endif
endif

#LDFLAGS = -lglfw -lvulkan -lGL -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

CXXFLAGS = -Wall -Wextra -std=c++17 -Iinclude

all:
	g++ $(CXXFLAGS) $(LIBS) ./src/main.cpp -o vulkan-proj 

run: all
	./vulkan-proj

clean:
	ifeq (WIN)
		del main
	else
		rm -f main
	endif
