#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inColor;
layout(location = 4) in uvec4 inBoneIndices;
layout(location = 5) in vec4 inBoneWeights;

struct InstanceData
{
    mat4 model;
    vec4 boundsSphere;
    uint firstIndex;
    uint indexCount;
    int vertexOffset;
    uint materialId;
    uint boneOffset;
    uint flags;
    uint pad0;
    uint pad1;
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

layout(std430, set = 0, binding = 2) readonly buffer InstanceBuffer
{
    InstanceData instances[];
};

layout(std430, set = 0, binding = 8) readonly buffer BonePaletteBuffer
{
    mat4 bones[];
};

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outWorldNormal;
layout(location = 2) out vec2 outUV;
layout(location = 3) out vec4 outColor;
layout(location = 4) out float outViewDepth;
layout(location = 5) flat out uint outMaterialId;

void main()
{
    InstanceData inst = instances[gl_InstanceIndex];
    vec4 localPos = vec4(inPosition, 1.0);
    vec3 localNorm = inNormal;

    if (inst.boneOffset != 0xFFFFFFFFu)
    {
        float wSum = inBoneWeights.x + inBoneWeights.y + inBoneWeights.z + inBoneWeights.w;
        if (wSum > 0.001)
        {
            mat4 skin = bones[inst.boneOffset + inBoneIndices.x] * inBoneWeights.x
                + bones[inst.boneOffset + inBoneIndices.y] * inBoneWeights.y
                + bones[inst.boneOffset + inBoneIndices.z] * inBoneWeights.z
                + bones[inst.boneOffset + inBoneIndices.w] * inBoneWeights.w;
            localPos = skin * localPos;
            localNorm = mat3(skin) * localNorm;
        }
    }

    vec4 worldPos = inst.model * localPos;
    vec4 viewPos = frame.view * worldPos;
    gl_Position = frame.proj * viewPos;

    outWorldPos = worldPos.xyz;
    outWorldNormal = normalize(mat3(inst.model) * localNorm);
    outUV = inUV;
    outColor = inColor;
    outViewDepth = -viewPos.z;
    outMaterialId = inst.materialId;
}
