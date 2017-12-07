//
//  utilities.cpp
//  OpengGL_MVP
//
//  Created by Muhammed Abdullah Al Ahad on 11/27/17.
//  Copyright © 2017 Muhammed Abdullah Al Ahad. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <sstream>
#include "utilities.hpp"



using namespace std;
using namespace glt;


// Adding call back method for the window
static void error_callback(int error, const char* description){
    cout<<"Error: "<<description<<endl;
}

static void framebuffer_size_callback(GLFWwindow* window, int width, int height){
    glViewport(0,0,width, height);
    cout<<"New buffer size : "<< width <<" X "<< height <<endl;
}

static void key_callback(GLFWwindow* window, int key , int scancode, int action, int mods){
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS){
        cout<<"Esc has been pressed..."<<endl;
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

GLFWwindow* utilities::createWindow(const char* title){
    cout<<"Creating GLFW window..."<<endl;


    GLFWwindow* window = NULL;


    // Adding error call back method
    glfwSetErrorCallback(error_callback);

    // Inittialize glfw
    if(! glfwInit()){
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // Setting GLFW window hints
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    window = glfwCreateWindow(WIDTH, HEIGHT, title, nullptr, nullptr);
    if(!window){
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // make the created opengl current context
    glfwMakeContextCurrent(window);
    //------------------------------------ Set Call back -------------------------------
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetKeyCallback(window, key_callback);

    //------------------------------- Load opengl function pointer -----------------------
    if(!gladLoadGL()){
      cout<<"Failed to get opengl function pointer..."<<endl;
    }


    glfwSwapInterval(1);

    return window;
}


// Create the shader program


struct ShaderProgramSource{
    std::string vertexSource;
    std::string fragmentSource;
};

static ShaderProgramSource parseShader(const std::string& filePath){
    std::ifstream myStream(filePath);

    enum class ShaderType
    {
        NONE = -1, VERTEX = 0, FRAGMENT = 1
    };

    ShaderType  sType = static_cast<ShaderType>(ShaderType::NONE);
    std::string line;
    std::stringstream sStream[2];

    if(myStream.is_open()) {
        while (getline(myStream, line)) {
            if (line.find("#Shader") != std::string::npos) {
                if (line.find("Vertex") != std::string::npos) {
                    sType = static_cast<ShaderType>(ShaderType::VERTEX);
                } else if (line.find("Fragment") != std::string::npos) {
                    sType = static_cast<ShaderType>(ShaderType::FRAGMENT);
                }
            } else {
                sStream[(int) sType] << line << "\n";
            }
        }
    } else{
        std::cout<<"File could not be opened...!!"<<std::endl;
    }
    return {sStream[0].str(),sStream[1].str()};
}

static unsigned  int createNcompileShader(const unsigned int shaderType, const std::string& shaderSource){
    unsigned int shader = glCreateShader(shaderType);
    const char* src = shaderSource.c_str();
    glShaderSource(shader,1,&src,NULL);
    glCompileShader(shader);

    // Error checking
    int success;
    char infoLog[512];
    glGetShaderiv(shader,GL_COMPILE_STATUS,&success);
    if(!success){
        glGetShaderInfoLog(shader,512,NULL,infoLog);
        std::cout<<"Error:: SHADER:: "<<shaderType<<" :: COMPILATION:: FAILED.\n"<<infoLog <<std::endl;
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

unsigned int utilities::createShaderProgram(const std::string filePath){
    unsigned int shaderProgram = glCreateProgram();

    ShaderProgramSource source = parseShader(filePath);


    unsigned int vertexShader = createNcompileShader(GL_VERTEX_SHADER, source.vertexSource);
    unsigned int fragmentShader = createNcompileShader(GL_FRAGMENT_SHADER,source.fragmentSource);

    // link the shaders to the shader program
    glAttachShader(shaderProgram,vertexShader);
    glAttachShader(shaderProgram,fragmentShader);
    glLinkProgram(shaderProgram);
    glValidateProgram(shaderProgram);

    // We can delete the shaders now as they are already linked attached to the shader program
    glDeleteShader(vertexShader);
    glDeleteShader(vertexShader);

    return shaderProgram;
}
