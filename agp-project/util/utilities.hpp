//
//  utilities.hpp
//  OpengGL_MVP
//
//  Created by Muhammed Abdullah Al Ahad on 11/27/17.
//  Copyright © 2017 Muhammed Abdullah Al Ahad. All rights reserved.
//

#ifndef UTILITIES_CPP
#define UTILITIES_CPP

#include <iostream>
#include "common.hpp"

namespace  glt{
    namespace utilities{
        // this function creates the GLFW window
        GLFWwindow* createWindow(const char* title);
        unsigned int createShaderProgram(const std::string filePath);
    }
}

#endif /* UTILITIES_CPP */
