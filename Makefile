LDFLAGS = -lglfw -lvulkan -lGL -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

all:
	g++ -std=c++17 ./src/main.cpp -o vulkan-proj $(LDFLAGS)

clean:
	rm -f vulkan-proj

run: all
	./vulkan-proj
