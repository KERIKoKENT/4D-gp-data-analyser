#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;

uniform vec3 lightPos;
uniform vec3 objectColor;

void main() {
    // 1. Градиент высоты (от синего к красному)
    // Масштабируем высоту под диапазон [0, 1] для смешивания цветов
    float h = clamp(FragPos.y * 0.5 + 0.5, 0.0, 1.0); 
    vec3 lowColor = vec3(0.1, 0.2, 0.8);  // Темно-синий
    vec3 highColor = vec3(0.9, 0.1, 0.1); // Ярко-красный
    vec3 baseColor = mix(lowColor, highColor, h);

    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    
    float ambientStrength = 0.3;
    float lightIntensity = 1.6;
    
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * vec3(1.0) * lightIntensity;
    
    vec3 result = (ambientStrength + diffuse) * baseColor;
    FragColor = vec4(result, 1.0);
}