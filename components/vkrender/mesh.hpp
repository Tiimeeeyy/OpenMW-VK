#ifndef OPENMW_COMPONENTS_VKRENDER_MESH_HPP
#define OPENMW_COMPONENTS_VKRENDER_MESH_HPP

#include <components/esm3/loadland.hpp>
#include <components/nif/niffile.hpp>

#include <osg/Vec3f>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace VkRender
{
    struct Vertex
    {
        float mPos[3];
        float mNormal[3];
        float mUV[2];
        float mColor[4];
        std::uint32_t mBoneIndices[4]{ 0, 0, 0, 0 };
        float mBoneWeights[4]{ 0.f, 0.f, 0.f, 0.f };
    };

    struct MaterialDesc
    {
        std::string mDiffuseTexture;
        float mBaseColor[4]{ 1.f, 1.f, 1.f, 1.f };
        bool mAlphaTest{ false };
        float mAlphaCutoff{ 0.5f };
        bool mIsWater{ false };
        bool mIsTransparent{ false };
    };

    struct Submesh
    {
        std::uint32_t mFirstIndex{ 0 };
        std::uint32_t mIndexCount{ 0 };
        std::uint32_t mMaterialIndex{ 0 };
        osg::Vec3f mBoundsCenter{ 0.f, 0.f, 0.f };
        float mBoundsRadius{ 1.f };
    };

    struct MeshData
    {
        std::vector<Vertex> mVertices;
        std::vector<std::uint32_t> mIndices;
        std::vector<Submesh> mSubmeshes;
        std::vector<MaterialDesc> mMaterials;
        osg::Vec3f mBoundsMin{ 0.f, 0.f, 0.f };
        osg::Vec3f mBoundsMax{ 0.f, 0.f, 0.f };
        bool mHasBounds{ false };
        std::size_t mSubmeshCount{ 0 };
    };

    /// Walk a parsed Nif::NIFFile and extract visible triangle submeshes + diffuse material paths.
    MeshData extractNifMesh(Nif::FileView file);

#ifdef OPENMW_VKRENDER_USE_ASSIMP
    /// Parse a mesh (e.g. .dae, .gltf, .glb, .obj) from memory using Assimp and extract submeshes + materials.
    MeshData extractAssimpMesh(std::span<const std::byte> data, std::string_view hintExtension);
#endif

    /// Build a chunked 65x65 terrain mesh from an ESM::Land record, grouped by VTEX/LTEX textures.
    MeshData buildTerrainMesh(
        const ESM::Land& land, const std::unordered_map<std::uint16_t, std::string>& landTextures = {});

    /// Build a flat water plane mesh centered at (centerX, centerY, waterHeight) with given extent.
    MeshData buildWaterPlaneMesh(float centerX, float centerY, float waterHeight, float size);
}

#endif
