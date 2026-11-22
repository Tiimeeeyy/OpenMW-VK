#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outFragColor;

layout(set = 0, binding = 0) uniform FrameData
{
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    mat4 invProj;
    mat4 cascadeViewProj[3];
    vec4 cascadeSplits;
    vec4 cameraPos;
    vec4 sunDir;
    vec4 sunColor;
    vec4 ambientColor;
    vec4 fogColor;
    vec4 fogParams;
    vec4 screenParams;
    vec4 visualParams; // x = skyEnabled, y = gtaoEnabled, z = bloomEnabled, w = toneMapEnabled
    vec4 frustumPlanes[6];
} frame;

layout(set = 1, binding = 0) uniform sampler2D gSceneColor;
layout(set = 1, binding = 1) uniform sampler2D gBloom;
layout(set = 1, binding = 2) uniform sampler2D gAO;

vec3 acesFilm(vec3 x)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main()
{
    vec3 color = texture(gSceneColor, inUV).rgb;

    // Subtle spatial 3x3 bilateral-style blur on half-res GTAO
    if (frame.visualParams.y > 0.5)
    {
        vec2 aoTexel = 1.0 / vec2(textureSize(gAO, 0));
        float ao = texture(gAO, inUV).r * 0.4;
        ao += texture(gAO, inUV + vec2(aoTexel.x, 0.0)).r * 0.15;
        ao += texture(gAO, inUV - vec2(aoTexel.x, 0.0)).r * 0.15;
        ao += texture(gAO, inUV + vec2(0.0, aoTexel.y)).r * 0.15;
        ao += texture(gAO, inUV - vec2(0.0, aoTexel.y)).r * 0.15;
        color *= mix(1.0, ao, 0.65);
    }

    if (frame.visualParams.z > 0.5)
    {
        vec3 bloom = texture(gBloom, inUV).rgb;
        color += bloom * 0.32;
    }

    // Subtle Contrast-Adaptive Sharpening (CAS) in linear space
    vec2 texel = 1.0 / max(frame.screenParams.xy, vec2(1.0));
    vec3 n = texture(gSceneColor, inUV + vec2(0.0, -texel.y)).rgb;
    vec3 s = texture(gSceneColor, inUV + vec2(0.0, texel.y)).rgb;
    vec3 e = texture(gSceneColor, inUV + vec2(texel.x, 0.0)).rgb;
    vec3 w = texture(gSceneColor, inUV + vec2(-texel.x, 0.0)).rgb;
    vec3 minN = min(color, min(min(n, s), min(e, w)));
    vec3 maxN = max(color, max(max(n, s), max(e, w)));
    vec3 sharp = color + (color - (n + s + e + w) * 0.25) * 0.18;
    color = clamp(sharp, minN, maxN);

    if (frame.visualParams.w > 0.5)
        color = acesFilm(color * 1.05);
    else
        color = clamp(color, 0.0, 1.0);

    // Subtle filmic vignette
    vec2 centered = inUV * 2.0 - 1.0;
    float vig = 1.0 - dot(centered, centered) * 0.08;
    color *= clamp(vig, 0.85, 1.0);

    outFragColor = vec4(color, 1.0);
}
