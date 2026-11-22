#ifndef OPENMW_COMPONENTS_VKRENDER_SCENE_HPP
#define OPENMW_COMPONENTS_VKRENDER_SCENE_HPP

#include <osg/Matrixf>
#include <osg/Vec3f>
#include <osg/Vec4f>

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace VkRender
{
    using TextureId = std::uint32_t;
    using MaterialId = std::uint32_t;
    using MeshId = std::uint32_t;
    using InstanceId = std::uint32_t;
    using LightId = std::uint32_t;

    inline constexpr std::uint32_t sInvalidHandle = UINT32_MAX;
    inline constexpr std::uint32_t sShadowCascadeCount = 3;

    enum MaterialFlags : std::uint32_t
    {
        MaterialFlag_None = 0u,
        MaterialFlag_AlphaTest = 1u << 0,
        MaterialFlag_Water = 1u << 1,
        MaterialFlag_Unlit = 1u << 2,
        MaterialFlag_Transparent = 1u << 3,
    };

    enum InstanceFlags : std::uint32_t
    {
        InstanceFlag_None = 0u,
        InstanceFlag_Transparent = 1u << 0,
    };

    /// Matches std430 MaterialData in mesh.frag
    struct alignas(16) GpuMaterial
    {
        float mBaseColor[4]{ 1.f, 1.f, 1.f, 1.f };
        std::uint32_t mAlbedoTexId{ 0 };
        std::uint32_t mNormalTexId{ 0 };
        std::uint32_t mFlags{ MaterialFlag_None };
        float mAlphaCutoff{ 0.5f };
        float mRoughness{ 0.80f };
        float mMetallic{ 0.0f };
        float mPad0{ 0.0f };
        float mPad1{ 0.0f };
    };

    /// Matches std430 InstanceData in cull.comp / shadow.vert / mesh.vert
    struct alignas(16) GpuInstance
    {
        float mModel[16]{};
        float mBoundsSphere[4]{ 0.f, 0.f, 0.f, 1.f };
        std::uint32_t mFirstIndex{ 0 };
        std::uint32_t mIndexCount{ 0 };
        std::int32_t mVertexOffset{ 0 };
        std::uint32_t mMaterialId{ 0 };
        std::uint32_t mBoneOffset{ sInvalidHandle };
        std::uint32_t mFlags{ InstanceFlag_None };
        std::uint32_t mPad0{ 0 };
        std::uint32_t mPad1{ 0 };
    };

    /// Matches std430 PointLight in light_cull.comp / mesh.frag
    struct alignas(16) GpuPointLight
    {
        float mPosRadius[4]{ 0.f, 0.f, 0.f, 256.f };
        float mColorIntensity[4]{ 1.f, 0.9f, 0.7f, 1.f };
    };

    /// Matches std430 TileLightList in light_cull.comp / mesh.frag (256 bytes per 16x16 screen tile)
    struct alignas(16) GpuTileLightList
    {
        std::uint32_t mCount{ 0 };
        std::uint32_t mIndices[63]{};
    };

    /// Matches std140 FrameData in all shaders
    struct alignas(16) FrameUniforms
    {
        float mView[16]{};
        float mProj[16]{};
        float mViewProj[16]{};
        float mInvProj[16]{};
        float mCascadeViewProj[sShadowCascadeCount][16]{};
        float mCascadeSplits[4]{ 600.f, 2400.f, 10000.f, 0.f };
        float mCameraPos[4]{ 0.f, 0.f, 0.f, 0.f };
        float mSunDir[4]{ 0.45f, -0.55f, 0.70f, 0.002f };
        float mSunColor[4]{ 1.0f, 0.95f, 0.86f, 1.25f };
        float mAmbientColor[4]{ 0.36f, 0.42f, 0.52f, 0.42f };
        float mFogColor[4]{ 0.58f, 0.66f, 0.76f, 10000.f };
        float mFogParams[4]{ 28000.f, 0.f, 32000.f, 1.f };
        float mScreenParams[4]{ 1280.f, 720.f, 80.f, 45.f };
        float mVisualParams[4]{ 1.f, 1.f, 1.f, 1.f }; // x=sky, y=gtao, z=bloom, w=toneMap
        float mFrustumPlanes[6][4]{};
    };

    struct FrameView
    {
        osg::Matrixf mView;
        osg::Matrixf mProj;
        osg::Matrixf mViewProj;
        std::array<osg::Matrixf, sShadowCascadeCount> mCascadeViewProj;
        osg::Vec3f mCascadeSplits{ 600.f, 2400.f, 10000.f };
        osg::Vec3f mCameraPos{ 0.f, 0.f, 0.f };
        float mTime{ 0.f };
        osg::Vec3f mSunDir{ 0.45f, -0.55f, 0.70f };
        osg::Vec4f mSunColor{ 1.0f, 0.95f, 0.86f, 1.25f };
        osg::Vec4f mAmbientColor{ 0.36f, 0.42f, 0.52f, 0.42f };
        osg::Vec3f mFogColor{ 0.58f, 0.66f, 0.76f };
        float mFogStart{ 10000.f };
        float mFogEnd{ 28000.f };
        float mMaxDrawDistance{ 32000.f };
        bool mEnableShadows{ true };
        bool mEnableFrustumCull{ true };
        bool mEnableSky{ true };
        bool mEnableGTAO{ true };
        bool mEnableBloom{ true };
        bool mEnableToneMap{ true };
    };

    inline void extractFrustumPlanes(const osg::Matrixf& vp, float outPlanes[6][4])
    {
        auto setPlane = [&](int idx, float a, float b, float c, float d) {
            const float len = std::sqrt(a * a + b * b + c * c);
            const float inv = (len > 1e-6f) ? (1.0f / len) : 1.0f;
            outPlanes[idx][0] = a * inv;
            outPlanes[idx][1] = b * inv;
            outPlanes[idx][2] = c * inv;
            outPlanes[idx][3] = d * inv;
        };

        setPlane(0, vp(0, 3) + vp(0, 0), vp(1, 3) + vp(1, 0), vp(2, 3) + vp(2, 0), vp(3, 3) + vp(3, 0));
        setPlane(1, vp(0, 3) - vp(0, 0), vp(1, 3) - vp(1, 0), vp(2, 3) - vp(2, 0), vp(3, 3) - vp(3, 0));
        setPlane(2, vp(0, 3) + vp(0, 1), vp(1, 3) + vp(1, 1), vp(2, 3) + vp(2, 1), vp(3, 3) + vp(3, 1));
        setPlane(3, vp(0, 3) - vp(0, 1), vp(1, 3) - vp(1, 1), vp(2, 3) - vp(2, 1), vp(3, 3) - vp(3, 1));
        setPlane(4, vp(0, 2), vp(1, 2), vp(2, 2), vp(3, 2));
        setPlane(5, vp(0, 3) - vp(0, 2), vp(1, 3) - vp(1, 2), vp(2, 3) - vp(2, 2), vp(3, 3) - vp(3, 2));
    }
}

#endif
