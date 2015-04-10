CXX ?= g++
CXXFLAGS = -std=c++11 -fomit-frame-pointer -fno-rtti -fno-exceptions
DEPS = smbf.h
OBJS = smbf.o

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

smbf: $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS)

clean:
	rm -f smbf $(OBJS)
