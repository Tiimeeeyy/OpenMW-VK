#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inWorldNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inColor;
layout(location = 4) in float inViewDepth;
layout(location = 5) flat in uint inMaterialId;

const float PI = 3.14159265359;

struct MaterialData
{
    vec4 baseColor;
    uint albedoTexId;
    uint normalTexId;
    uint flags; // bit 0 = alphaTest, bit 1 = water, bit 2 = unlit, bit 3 = transparent
    float alphaCutoff;
    float roughness;
    float metallic;
    float pad0;
    float pad1;
};

struct PointLight
{
    vec4 posRadius;
    vec4 colorIntensity;
};

struct TileLightList
{
    uint count;
    uint indices[63];
};

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

layout(std430, set = 0, binding = 1) readonly buffer MaterialBuffer
{
    MaterialData materials[];
};

layout(std430, set = 0, binding = 3) readonly buffer LightBuffer
{
    PointLight lights[];
};

layout(set = 0, binding = 4) uniform sampler gSampler;
layout(set = 0, binding = 5) uniform texture2D gTextures[];
layout(set = 0, binding = 6) uniform sampler2DArrayShadow gShadowMap;

layout(std430, set = 0, binding = 7) readonly buffer TileLightBuffer
{
    TileLightList tiles[];
};

layout(set = 0, binding = 9) uniform sampler2D gSceneDepth;

layout(location = 0) out vec4 outFragColor;

// 9-tap rotated Vogel-disk PCF on 3-Cascade Shadow Map
float sampleCascadedShadow(vec3 worldPos, vec3 n, vec3 l)
{
    if (frame.fogParams.w < 0.5)
        return 1.0;

    int cascade = 2;
    if (inViewDepth < frame.cascadeSplits.x)
        cascade = 0;
    else if (inViewDepth < frame.cascadeSplits.y)
        cascade = 1;

    vec4 lightClip = frame.cascadeViewProj[cascade] * vec4(worldPos, 1.0);
    vec3 proj = lightClip.xyz / lightClip.w;
    vec2 uv = proj.xy * 0.5 + 0.5;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z < 0.0 || proj.z > 1.0)
        return 1.0;

    float biasScale = (cascade == 0) ? 0.65 : ((cascade == 1) ? 1.15 : 2.0);
    float bias = max(frame.sunDir.w * biasScale * (1.0 - dot(n, l)), frame.sunDir.w * 0.25 * biasScale);
    float depth = proj.z - bias;
    vec2 texelSize = 1.0 / vec2(textureSize(gShadowMap, 0).xy);

    float angle = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453) * 6.2831853;
    float sa = sin(angle);
    float ca = cos(angle);
    mat2 rot = mat2(ca, -sa, sa, ca);

    float shadow = 0.0;
    const int TAPS = 9;
    for (int i = 0; i < TAPS; ++i)
    {
        float r = sqrt((float(i) + 0.5) / float(TAPS)) * 1.85;
        float theta = float(i) * 2.39996323;
        vec2 offset = (rot * (vec2(cos(theta), sin(theta)) * r)) * texelSize;
        shadow += texture(gShadowMap, vec4(uv + offset, float(cascade), depth));
    }
    return shadow / float(TAPS);
}

// Cotangent-frame normal mapping (works on any mesh without precomputed tangents)
vec3 perturbNormal(vec3 N, vec3 V, vec2 uv, vec3 mapN)
{
    vec3 dp1 = dFdx(inWorldPos);
    vec3 dp2 = dFdy(inWorldPos);
    vec2 duv1 = dFdx(uv);
    vec2 duv2 = dFdy(uv);

    vec3 dp2perp = cross(dp2, N);
    vec3 dp1perp = cross(N, dp1);
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;

    float invmax = inversesqrt(max(dot(T, T), dot(B, B)) + 1e-6);
    mat3 TBN = mat3(T * invmax, B * invmax, N);
    return normalize(TBN * mapN);
}

// Cook-Torrance GGX PBR Direct Lighting
vec3 evalPBR(vec3 N, vec3 V, vec3 L, vec3 albedo, float roughness, float metallic, vec3 radiance)
{
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0)
        return vec3(0.0);

    vec3 H = normalize(V + L);
    float NdotV = max(dot(N, V), 0.001);
    float NdotH = max(dot(N, H), 0.0);
    float HdotV = max(dot(H, V), 0.0);

    float a = roughness * roughness;
    float a2 = a * a;
    float denom = (NdotH * NdotH * (a2 - 1.0) + 1.0);
    float D = a2 / max(PI * denom * denom, 1e-5);

    float k = (roughness + 1.0);
    k = (k * k) / 8.0;
    float gV = NdotV / (NdotV * (1.0 - k) + k);
    float gL = NdotL / (NdotL * (1.0 - k) + k);
    float G = gV * gL;

    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = F0 + (1.0 - F0) * pow(clamp(1.0 - HdotV, 0.0, 1.0), 5.0);

    vec3 spec = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);
    vec3 kD = (1.0 - F) * (1.0 - metallic);
    return (kD * albedo / PI + spec) * radiance * NdotL;
}

float linearizeDepth(float depth, vec2 screenUV)
{
    vec2 ndc = screenUV * 2.0 - 1.0;
    vec4 v = frame.invProj * vec4(ndc, depth, 1.0);
    return -v.z / v.w;
}

void main()
{
    MaterialData mat = materials[inMaterialId];
    vec4 texSample = texture(sampler2D(gTextures[nonuniformEXT(mat.albedoTexId)], gSampler), inUV);

    if ((mat.flags & 1u) != 0u && texSample.a < mat.alphaCutoff)
        discard;

    vec3 n = normalize(inWorldNormal);
    if (!gl_FrontFacing)
        n = -n;

    vec3 v = normalize(frame.cameraPos.xyz - inWorldPos);
    if (mat.normalTexId > 0u)
    {
        vec3 mapN = texture(sampler2D(gTextures[nonuniformEXT(mat.normalTexId)], gSampler), inUV).xyz * 2.0 - 1.0;
        n = perturbNormal(n, v, inUV, mapN);
    }

    vec3 albedo = texSample.rgb * mat.baseColor.rgb * inColor.rgb;
    float alpha = texSample.a * mat.baseColor.a * inColor.a;
    float roughness = clamp(mat.roughness, 0.08, 0.98);
    float metallic = clamp(mat.metallic, 0.0, 1.0);

    vec3 sunL = normalize(frame.sunDir.xyz);
    float shadow = sampleCascadedShadow(inWorldPos, n, sunL);
    vec3 sunRadiance = frame.sunColor.rgb * frame.sunColor.w * 2.2;

    if ((mat.flags & 2u) != 0u)
    {
        // Depth-Aware Stylized PBR Water: Beer-Lambert absorption + shoreline foam + sky reflection + GGX sun glint
        vec2 screenUV = gl_FragCoord.xy / max(frame.screenParams.xy, vec2(1.0));
        float opaqueDepthRaw = texture(gSceneDepth, screenUV).r;
        float opaqueLinear = linearizeDepth(opaqueDepthRaw, screenUV);
        float waterDepth = max(opaqueLinear - inViewDepth, 0.0);

        float t = frame.cameraPos.w;
        vec2 p = inWorldPos.xy * 0.018;
        float w1 = sin(p.x + t * 1.3) * cos(p.y * 0.9 - t * 1.0);
        float w2 = sin((p.x * 1.6 - p.y * 1.3) + t * 1.8) * 0.5;
        float w3 = cos((p.x * 2.7 + p.y * 2.1) - t * 2.4) * 0.25;
        n = normalize(vec3((w1 + w2 + w3) * 0.055, (-w1 + w2 - w3) * 0.055, 1.0));

        float fresnel = 0.02 + 0.98 * pow(clamp(1.0 - max(dot(n, v), 0.0), 0.0, 1.0), 5.0);
        vec3 shallowColor = vec3(0.08, 0.34, 0.38);
        vec3 deepColor = vec3(0.02, 0.09, 0.16);
        float depthFactor = 1.0 - exp(-waterDepth * 0.006);
        vec3 waterBody = mix(shallowColor, deepColor, depthFactor);

        vec3 reflDir = reflect(-v, n);
        float skyGrad = clamp(reflDir.z, 0.0, 1.0);
        vec3 skyRefl = mix(frame.fogColor.rgb * 1.1, vec3(0.18, 0.38, 0.72), pow(skyGrad, 0.6));

        vec3 H = normalize(sunL + v);
        float sunSpec = pow(max(dot(n, H), 0.0), 220.0) * 3.5 * shadow;

        // Animated shoreline foam where water meets terrain
        float foamWave = sin(waterDepth * 0.28 - t * 2.2) * 0.5 + 0.5;
        float foam = (1.0 - smoothstep(0.0, 28.0, waterDepth)) * (0.45 + 0.55 * foamWave);

        vec3 waterLit = mix(waterBody, skyRefl, clamp(fresnel * 1.15, 0.0, 0.92)) + frame.sunColor.rgb * sunSpec;
        waterLit = mix(waterLit, vec3(0.86, 0.92, 0.96), foam * 0.72);
        alpha = clamp(mix(0.28, 0.94, depthFactor) + fresnel * 0.35 + foam * 0.5, 0.0, 1.0);

        float fogStart = frame.fogColor.w;
        float fogEnd = frame.fogParams.x;
        if (fogEnd > fogStart)
        {
            float dist = length(inWorldPos - frame.cameraPos.xyz);
            float fogFactor = clamp((dist - fogStart) / (fogEnd - fogStart), 0.0, 1.0);
            waterLit = mix(waterLit, frame.fogColor.rgb, fogFactor);
        }
        outFragColor = vec4(waterLit, alpha);
        return;
    }

    // Hemisphere IBL ambient
    float hemi = clamp(n.z * 0.5 + 0.5, 0.0, 1.0);
    vec3 skyAmb = frame.ambientColor.rgb;
    vec3 groundAmb = frame.ambientColor.rgb * frame.ambientColor.w * vec3(0.95, 0.90, 0.80);
    vec3 ambient = albedo * mix(groundAmb, skyAmb, hemi);

    // Direct PBR sun
    vec3 direct = evalPBR(n, v, sunL, albedo, roughness, metallic, sunRadiance) * shadow;

    // Forward+ Tiled Point Lights (PBR)
    uint tileCountX = uint(frame.screenParams.z);
    uint tileCountY = uint(frame.screenParams.w);
    uint tx = min(uint(gl_FragCoord.x) / 16u, max(tileCountX, 1u) - 1u);
    uint ty = min(uint(gl_FragCoord.y) / 16u, max(tileCountY, 1u) - 1u);
    uint tileIdx = ty * tileCountX + tx;
    uint tileLightCount = min(tiles[tileIdx].count, 63u);

    for (uint i = 0u; i < tileLightCount; ++i)
    {
        uint lightIdx = tiles[tileIdx].indices[i];
        vec3 toLight = lights[lightIdx].posRadius.xyz - inWorldPos;
        float dist = length(toLight);
        float radius = lights[lightIdx].posRadius.w;
        if (dist < radius && dist > 1e-4)
        {
            vec3 l = toLight / dist;
            float atten = clamp(1.0 - (dist / radius), 0.0, 1.0);
            atten = atten * atten;
            vec3 lRad = lights[lightIdx].colorIntensity.rgb * lights[lightIdx].colorIntensity.w * 2.5 * atten;
            direct += evalPBR(n, v, l, albedo, roughness, metallic, lRad);
        }
    }

    vec3 lit = ((mat.flags & 4u) != 0u) ? albedo : (ambient + direct);

    // Exponential height-based atmospheric fog + subtle sun in-scattering
    float fogStart = frame.fogColor.w;
    float fogEnd = frame.fogParams.x;
    if (fogEnd > fogStart)
    {
        float dist = length(inWorldPos - frame.cameraPos.xyz);
        float distFog = clamp((dist - fogStart) / (fogEnd - fogStart), 0.0, 1.0);
        float heightAtten = exp(-max(inWorldPos.z, 0.0) * 0.00035);
        float fogFactor = clamp(distFog * (0.45 + 0.55 * heightAtten), 0.0, 1.0);
        float sunScatter = pow(max(dot(-v, sunL), 0.0), 6.0) * 0.25;
        vec3 fogCol = frame.fogColor.rgb + frame.sunColor.rgb * sunScatter;
        lit = mix(lit, fogCol, fogFactor);
    }

    outFragColor = vec4(lit, ((mat.flags & 8u) != 0u) ? alpha : 1.0);
}
