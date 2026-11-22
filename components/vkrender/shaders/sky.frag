#version 450

layout(location = 0) in vec2 inNDC;
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
    vec4 visualParams;
    vec4 frustumPlanes[6];
} frame;

float hash21(vec2 p)
{
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float noise(vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p)
{
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 4; ++i)
    {
        v += a * noise(p);
        p = p * 2.04 + vec2(17.1, 31.7);
        a *= 0.5;
    }
    return v;
}

void main()
{
    vec4 viewRay = frame.invProj * vec4(inNDC, 1.0, 1.0);
    vec3 dirView = normalize(viewRay.xyz / viewRay.w);
    vec3 dirWorld = normalize(transpose(mat3(frame.view)) * dirView);

    float h = clamp(dirWorld.z, 0.0, 1.0);
    vec3 horizon = frame.fogColor.rgb * 1.15;
    vec3 midSky = vec3(0.28, 0.50, 0.82);
    vec3 zenith = vec3(0.09, 0.22, 0.54);
    vec3 sky = mix(horizon, midSky, smoothstep(0.0, 0.25, h));
    sky = mix(sky, zenith, smoothstep(0.18, 0.85, h));

    // Mie sun scattering + HDR sun disc (triggers subtle bloom)
    vec3 sunL = normalize(frame.sunDir.xyz);
    float sunCos = max(dot(dirWorld, sunL), 0.0);
    float mieGlow = pow(sunCos, 8.0) * 0.35 + pow(sunCos, 64.0) * 0.65;
    float sunDisc = smoothstep(0.9991, 0.9996, sunCos) * 4.5;
    sky += frame.sunColor.rgb * (mieGlow + sunDisc);

    // Subtle procedural high-altitude clouds
    if (dirWorld.z > 0.03)
    {
        vec2 cloudUV = (dirWorld.xy / (dirWorld.z + 0.15)) * 1.35 + vec2(frame.cameraPos.w * 0.008, -frame.cameraPos.w * 0.004);
        float c = fbm(cloudUV);
        float cloudMask = smoothstep(0.50, 0.78, c) * smoothstep(0.03, 0.22, dirWorld.z);
        vec3 cloudLit = mix(vec3(0.72, 0.78, 0.88), frame.sunColor.rgb * 1.35, 0.45 + 0.55 * pow(sunCos, 3.0));
        sky = mix(sky, cloudLit, cloudMask * 0.68);
    }

    outFragColor = vec4(sky, 1.0);
}
