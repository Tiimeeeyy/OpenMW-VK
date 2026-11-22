#version 450

layout(location = 0) in vec3 inPosition;
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

layout(push_constant) uniform ShadowPC
{
    uint cascadeIndex;
} pc;

void main()
{
    InstanceData inst = instances[gl_InstanceIndex];
    vec4 localPos = vec4(inPosition, 1.0);
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
        }
    }
    vec4 worldPos = inst.model * localPos;
    gl_Position = frame.cascadeViewProj[pc.cascadeIndex] * worldPos;
}
