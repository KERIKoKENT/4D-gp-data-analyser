#version 330 core

out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in float Confidence;

uniform vec3 lightPos;
uniform vec3 viewPos;

vec3 heatmap(float t)
{
    t = clamp(t, 0.0, 1.0);

    vec3 red = vec3(0.85, 0.2, 0.2);
    vec3 green = vec3(0.2, 0.85, 0.3);

    return mix(red, green, t);
}

void main()
{
    vec3 baseColor = heatmap(Confidence);

    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);

    float ambientStrength = 0.15;
    float diffuseStrength = 0.75;

    float diff = max(dot(norm, lightDir), 0.0);

    vec3 ambient = ambientStrength * baseColor;
    vec3 diffuse = diff * diffuseStrength * baseColor;

    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);

    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = spec * baseColor * 0.3;

    vec3 result = ambient + diffuse + specular;

    FragColor = vec4(result, 1.0);
}