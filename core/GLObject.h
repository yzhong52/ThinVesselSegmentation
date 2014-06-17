#ifndef GLOBJECT_H
#define GLOBJECT_H

namespace GLViewer
{

// derive the following virtual class in order to render your own objects
class Object
{
public:
    Object() {}
    virtual ~Object() {}

    ////////////////////////////////////////////////////
    // Pure virtual functions
    // you have to implement these virtual functions in order to render it
    // with GLViewer
    ////////////////////////////////////////////////////

    virtual void render(void) = 0;				 // render the object
    virtual unsigned int size_x(void) const = 0; // size of the object
    virtual unsigned int size_y(void) const = 0; // size of the object
    virtual unsigned int size_z(void) const = 0; // size of the object

    ////////////////////////////////////////////////////
    // Optional funtions to overide
    ////////////////////////////////////////////////////
    // init function for OpenGL, excuted before rendering loop
    virtual void init(void) { }

    // keyboard function for OpenGL
    virtual void keyboard( unsigned char key )
    {
        std::cout << "Key " << key << " is not defined for the object." << std::endl;
    }
};

}

#endif // GLOBJECT_H
