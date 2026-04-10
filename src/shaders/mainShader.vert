#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 view;
uniform mat4 projection;

out vec3 FragPos;
out vec3 Normal;

void main() {
    // Swap Y and Z axes for OpenGL
    vec3 swappedPos = vec3(aPos.x, aPos.z, aPos.y); 
    vec3 swappedNorm = vec3(aNormal.x, aNormal.z, aNormal.y);
    
    FragPos = swappedPos;
    Normal = normalize(swappedNorm);
    gl_Position = projection * view * vec4(swappedPos, 1.0);
}