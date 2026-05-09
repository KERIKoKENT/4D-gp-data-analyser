#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in float aConf;
layout(location = 2) in vec3 aNormal;

uniform mat4 view;
uniform mat4 projection;

out vec3 FragPos;
out vec3 Normal;
out float Confidence;

void main()
{
    vec3 swappedPos = vec3(aPos.x, aPos.z, aPos.y);
    vec3 swappedNorm = vec3(aNormal.x, aNormal.z, aNormal.y);

    FragPos = swappedPos;
    Normal = normalize(swappedNorm);
    Confidence = aConf;

    gl_Position = projection * view * vec4(swappedPos, 1.0);
}