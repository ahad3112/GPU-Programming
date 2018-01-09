//
//  utilities.hpp
//  OpengGL_MVP
//
//  Created by Muhammed Abdullah Al Ahad on 11/27/17.
//  Copyright © 2017 Muhammed Abdullah Al Ahad. All rights reserved.
//

#ifndef UTILITIES_CPP
#define UTILITIES_CPP
#include <glad/glad.h>
#include <GL/gl.h>
#include <GL/glx.h>
#include <GLFW/glfw3.h>
#include <iostream>

namespace  glt{
    namespace utilities{
        // this function creates the GLFW window
        GLFWwindow* createWindow(const char* title, int WIDTH, int HEIGHT);
        unsigned int createShaderProgram(const std::string filePath);
    }
}

#endif /* UTILITIES_CPP */
