#Shader Vertex
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 color;
uniform mat4 MVP;
out vec4 myColor;
void main()
{
    gl_Position = MVP * vec4(aPos,1.0);
    gl_PointSize = 2.0;
    myColor = vec4(color,0.5f);
}


#Shader Fragment
#version 330 core
out vec4 FragColor;
in vec4 myColor;
void main()
{
    FragColor = myColor;
}
