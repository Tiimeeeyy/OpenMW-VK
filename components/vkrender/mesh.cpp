#include "mesh.hpp"

#include <components/nif/data.hpp>
#include <components/nif/node.hpp>
#include <components/nif/property.hpp>
#include <components/nif/record.hpp>
#include <components/nif/texture.hpp>

#include <osg/Matrixf>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef OPENMW_VKRENDER_USE_ASSIMP
#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#endif

namespace VkRender
{
    namespace
    {
        void expandBounds(osg::Vec3f& bMin, osg::Vec3f& bMax, bool& hasBounds, const osg::Vec3f& p)
        {
            if (!hasBounds)
            {
                bMin = p;
                bMax = p;
                hasBounds = true;
                return;
            }
            bMin.x() = std::min(bMin.x(), p.x());
            bMin.y() = std::min(bMin.y(), p.y());
            bMin.z() = std::min(bMin.z(), p.z());
            bMax.x() = std::max(bMax.x(), p.x());
            bMax.y() = std::max(bMax.y(), p.y());
            bMax.z() = std::max(bMax.z(), p.z());
        }

        osg::Vec3f transformNormal(const osg::Vec3f& n, const osg::Matrixf& mat)
        {
            osg::Vec3f res = osg::Matrixf::transform3x3(n, mat);
            const float len = res.length();
            if (len > 1e-6f)
                res /= len;
            else
                res = osg::Vec3f(0.f, 0.f, 1.f);
            return res;
        }

        MaterialDesc extractNifMaterial(const Nif::NiAVObject* obj)
        {
            MaterialDesc mat;
            for (const auto& propPtr : obj->mProperties)
            {
                if (propPtr.empty())
                    continue;
                if (const auto* texProp = dynamic_cast<const Nif::NiTexturingProperty*>(propPtr.getPtr()))
                {
                    if (!texProp->mTextures.empty())
                    {
                        const auto& baseTex = texProp->mTextures[Nif::NiTexturingProperty::BaseTexture];
                        if (baseTex.mEnabled && !baseTex.mSourceTexture.empty())
                            mat.mDiffuseTexture = baseTex.mSourceTexture->mFile;
                    }
                }
                else if (const auto* alphaProp = dynamic_cast<const Nif::NiAlphaProperty*>(propPtr.getPtr()))
                {
                    if ((alphaProp->mFlags & Nif::NiAlphaProperty::Flag_Testing) != 0)
                    {
                        mat.mAlphaTest = true;
                        mat.mAlphaCutoff = static_cast<float>(alphaProp->mThreshold) / 255.0f;
                    }
                }
            }
            return mat;
        }

        void appendGeometry(MeshData& out, const Nif::NiGeometryData& data,
            const std::vector<std::uint32_t>& localIndices, const osg::Matrixf& worldMat, const MaterialDesc& matDesc)
        {
            if (data.mVertices.empty() || localIndices.empty())
                return;

            const std::size_t numVerts = data.mVertices.size();
            const std::uint32_t baseVertex = static_cast<std::uint32_t>(out.mVertices.size());

            osg::Vec3f subMin;
            osg::Vec3f subMax;
            bool subHasBounds = false;

            std::vector<osg::Vec3f> worldPos(numVerts);
            for (std::size_t i = 0; i < numVerts; ++i)
            {
                worldPos[i] = data.mVertices[i] * worldMat;
                expandBounds(out.mBoundsMin, out.mBoundsMax, out.mHasBounds, worldPos[i]);
                expandBounds(subMin, subMax, subHasBounds, worldPos[i]);
            }

            std::vector<osg::Vec3f> worldNorm(numVerts, osg::Vec3f(0.f, 0.f, 0.f));
            const bool hasNormals = data.mNormals.size() == numVerts;
            if (hasNormals)
            {
                for (std::size_t i = 0; i < numVerts; ++i)
                    worldNorm[i] = transformNormal(data.mNormals[i], worldMat);
            }
            else
            {
                for (std::size_t i = 0; i + 2 < localIndices.size(); i += 3)
                {
                    const std::uint32_t i0 = localIndices[i];
                    const std::uint32_t i1 = localIndices[i + 1];
                    const std::uint32_t i2 = localIndices[i + 2];
                    if (i0 >= numVerts || i1 >= numVerts || i2 >= numVerts)
                        continue;
                    const osg::Vec3f e1 = worldPos[i1] - worldPos[i0];
                    const osg::Vec3f e2 = worldPos[i2] - worldPos[i0];
                    const osg::Vec3f fn = e1 ^ e2;
                    worldNorm[i0] += fn;
                    worldNorm[i1] += fn;
                    worldNorm[i2] += fn;
                }
                for (osg::Vec3f& n : worldNorm)
                {
                    const float len = n.length();
                    if (len > 1e-6f)
                        n /= len;
                    else
                        n = osg::Vec3f(0.f, 0.f, 1.f);
                }
            }

            const bool hasUV = !data.mUVList.empty() && data.mUVList[0].size() == numVerts;
            const bool hasColors = data.mColors.size() == numVerts;

            out.mVertices.reserve(out.mVertices.size() + numVerts);
            for (std::size_t i = 0; i < numVerts; ++i)
            {
                Vertex v{};
                v.mPos[0] = worldPos[i].x();
                v.mPos[1] = worldPos[i].y();
                v.mPos[2] = worldPos[i].z();

                v.mNormal[0] = worldNorm[i].x();
                v.mNormal[1] = worldNorm[i].y();
                v.mNormal[2] = worldNorm[i].z();

                if (hasUV)
                {
                    v.mUV[0] = data.mUVList[0][i].x();
                    v.mUV[1] = data.mUVList[0][i].y();
                }

                if (hasColors)
                {
                    v.mColor[0] = data.mColors[i].r();
                    v.mColor[1] = data.mColors[i].g();
                    v.mColor[2] = data.mColors[i].b();
                    v.mColor[3] = data.mColors[i].a();
                }
                else
                {
                    v.mColor[0] = 1.f;
                    v.mColor[1] = 1.f;
                    v.mColor[2] = 1.f;
                    v.mColor[3] = 1.f;
                }

                out.mVertices.push_back(v);
            }

            const std::uint32_t firstIndex = static_cast<std::uint32_t>(out.mIndices.size());
            out.mIndices.reserve(out.mIndices.size() + localIndices.size());
            for (std::uint32_t idx : localIndices)
            {
                if (idx < numVerts)
                    out.mIndices.push_back(baseVertex + idx);
            }
            const std::uint32_t indexCount = static_cast<std::uint32_t>(out.mIndices.size()) - firstIndex;
            if (indexCount == 0)
                return;

            const std::uint32_t matIdx = static_cast<std::uint32_t>(out.mMaterials.size());
            out.mMaterials.push_back(matDesc);

            Submesh sub;
            sub.mFirstIndex = firstIndex;
            sub.mIndexCount = indexCount;
            sub.mMaterialIndex = matIdx;
            if (subHasBounds)
            {
                sub.mBoundsCenter = (subMin + subMax) * 0.5f;
                sub.mBoundsRadius = std::max(((subMax - subMin) * 0.5f).length(), 0.1f);
            }
            out.mSubmeshes.push_back(sub);
            ++out.mSubmeshCount;
        }

        void visitNode(MeshData& out, const Nif::NiAVObject* obj, const osg::Matrixf& parentMat)
        {
            if (obj == nullptr || obj->isHidden())
                return;

            if (obj->mRecordType == Nif::RC_RootCollisionNode || obj->mRecordType == Nif::RC_AvoidNode)
                return;

            const osg::Matrixf worldMat = obj->mTransform.toMatrix() * parentMat;

            if (const auto* triShape = dynamic_cast<const Nif::NiTriShape*>(obj))
            {
                if (!triShape->mData.empty())
                {
                    if (const auto* data = dynamic_cast<const Nif::NiTriShapeData*>(triShape->mData.getPtr()))
                    {
                        std::vector<std::uint32_t> indices;
                        indices.reserve(data->mTriangles.size());
                        for (unsigned short idx : data->mTriangles)
                            indices.push_back(idx);
                        appendGeometry(out, *data, indices, worldMat, extractNifMaterial(obj));
                    }
                }
                return;
            }

            if (const auto* triStrips = dynamic_cast<const Nif::NiTriStrips*>(obj))
            {
                if (!triStrips->mData.empty())
                {
                    if (const auto* data = dynamic_cast<const Nif::NiTriStripsData*>(triStrips->mData.getPtr()))
                    {
                        std::vector<std::uint32_t> indices;
                        for (const auto& strip : data->mStrips)
                        {
                            if (strip.size() < 3)
                                continue;
                            for (std::size_t i = 0; i + 2 < strip.size(); ++i)
                            {
                                const std::uint32_t a = strip[i];
                                const std::uint32_t b = (i % 2 == 0) ? strip[i + 1] : strip[i + 2];
                                const std::uint32_t c = (i % 2 == 0) ? strip[i + 2] : strip[i + 1];
                                if (a == b || b == c || a == c)
                                    continue;
                                indices.push_back(a);
                                indices.push_back(b);
                                indices.push_back(c);
                            }
                        }
                        appendGeometry(out, *data, indices, worldMat, extractNifMaterial(obj));
                    }
                }
                return;
            }

            if (const auto* node = dynamic_cast<const Nif::NiNode*>(obj))
            {
                for (const auto& child : node->mChildren)
                {
                    if (!child.empty())
                        visitNode(out, child.getPtr(), worldMat);
                }
            }
        }
    }

    MeshData extractNifMesh(Nif::FileView file)
    {
        MeshData result;
        const osg::Matrixf identity = osg::Matrixf::identity();
        for (std::size_t i = 0; i < file.numRoots(); ++i)
        {
            if (const auto* root = dynamic_cast<const Nif::NiAVObject*>(file.getRoot(i)))
                visitNode(result, root, identity);
        }
        return result;
    }

#ifdef OPENMW_VKRENDER_USE_ASSIMP
    MeshData extractAssimpMesh(std::span<const std::byte> data, std::string_view hintExtension)
    {
        Assimp::Importer importer;
        const unsigned int flags = aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices
            | aiProcess_PreTransformVertices | aiProcess_FlipUVs;

        std::string hint(hintExtension);
        if (!hint.empty() && hint.front() == '.')
            hint.erase(hint.begin());

        const aiScene* scene = importer.ReadFileFromMemory(data.data(), data.size(), flags, hint.c_str());
        if (scene == nullptr || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) != 0 || scene->mRootNode == nullptr)
            throw std::runtime_error(std::string("Assimp failed to load mesh: ") + importer.GetErrorString());

        MeshData out;
        out.mMaterials.resize(std::max(1u, scene->mNumMaterials));
        for (unsigned int i = 0; i < scene->mNumMaterials; ++i)
        {
            const aiMaterial* aiMat = scene->mMaterials[i];
            if (aiMat == nullptr)
                continue;
            MaterialDesc desc;
            aiString texPath;
            if (aiMat->GetTexture(aiTextureType_DIFFUSE, 0, &texPath) == AI_SUCCESS && texPath.length > 0)
                desc.mDiffuseTexture = texPath.C_Str();
            else if (aiMat->GetTexture(aiTextureType_BASE_COLOR, 0, &texPath) == AI_SUCCESS && texPath.length > 0)
                desc.mDiffuseTexture = texPath.C_Str();

            aiColor4D diffuse(1.f, 1.f, 1.f, 1.f);
            if (aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse) == AI_SUCCESS && desc.mDiffuseTexture.empty())
            {
                desc.mBaseColor[0] = diffuse.r;
                desc.mBaseColor[1] = diffuse.g;
                desc.mBaseColor[2] = diffuse.b;
                desc.mBaseColor[3] = diffuse.a;
            }
            out.mMaterials[i] = std::move(desc);
        }

        for (unsigned int m = 0; m < scene->mNumMeshes; ++m)
        {
            const aiMesh* mesh = scene->mMeshes[m];
            if (mesh == nullptr || mesh->mNumVertices == 0 || mesh->mNumFaces == 0)
                continue;

            const std::uint32_t baseVertex = static_cast<std::uint32_t>(out.mVertices.size());
            out.mVertices.reserve(out.mVertices.size() + mesh->mNumVertices);

            osg::Vec3f subMin;
            osg::Vec3f subMax;
            bool subHasBounds = false;

            const bool hasNormals = mesh->HasNormals();
            const bool hasUV = mesh->HasTextureCoords(0);
            const bool hasColor = mesh->HasVertexColors(0);

            for (unsigned int i = 0; i < mesh->mNumVertices; ++i)
            {
                const aiVector3D& p = mesh->mVertices[i];
                const osg::Vec3f wp(p.x, p.y, p.z);
                expandBounds(out.mBoundsMin, out.mBoundsMax, out.mHasBounds, wp);
                expandBounds(subMin, subMax, subHasBounds, wp);

                Vertex v{};
                v.mPos[0] = p.x;
                v.mPos[1] = p.y;
                v.mPos[2] = p.z;

                if (hasNormals)
                {
                    v.mNormal[0] = mesh->mNormals[i].x;
                    v.mNormal[1] = mesh->mNormals[i].y;
                    v.mNormal[2] = mesh->mNormals[i].z;
                }
                else
                {
                    v.mNormal[0] = 0.f;
                    v.mNormal[1] = 0.f;
                    v.mNormal[2] = 1.f;
                }

                if (hasUV)
                {
                    v.mUV[0] = mesh->mTextureCoords[0][i].x;
                    v.mUV[1] = mesh->mTextureCoords[0][i].y;
                }

                if (hasColor)
                {
                    v.mColor[0] = mesh->mColors[0][i].r;
                    v.mColor[1] = mesh->mColors[0][i].g;
                    v.mColor[2] = mesh->mColors[0][i].b;
                    v.mColor[3] = mesh->mColors[0][i].a;
                }
                else
                {
                    v.mColor[0] = 1.f;
                    v.mColor[1] = 1.f;
                    v.mColor[2] = 1.f;
                    v.mColor[3] = 1.f;
                }

                out.mVertices.push_back(v);
            }

            const std::uint32_t firstIndex = static_cast<std::uint32_t>(out.mIndices.size());
            out.mIndices.reserve(out.mIndices.size() + mesh->mNumFaces * 3);
            for (unsigned int f = 0; f < mesh->mNumFaces; ++f)
            {
                const aiFace& face = mesh->mFaces[f];
                if (face.mNumIndices != 3)
                    continue;
                out.mIndices.push_back(baseVertex + face.mIndices[0]);
                out.mIndices.push_back(baseVertex + face.mIndices[1]);
                out.mIndices.push_back(baseVertex + face.mIndices[2]);
            }
            const std::uint32_t indexCount = static_cast<std::uint32_t>(out.mIndices.size()) - firstIndex;
            if (indexCount == 0)
                continue;

            Submesh sub;
            sub.mFirstIndex = firstIndex;
            sub.mIndexCount = indexCount;
            sub.mMaterialIndex = std::min(mesh->mMaterialIndex, static_cast<unsigned int>(out.mMaterials.size() - 1));
            if (subHasBounds)
            {
                sub.mBoundsCenter = (subMin + subMax) * 0.5f;
                sub.mBoundsRadius = std::max(((subMax - subMin) * 0.5f).length(), 0.1f);
            }
            out.mSubmeshes.push_back(sub);
            ++out.mSubmeshCount;
        }
        return out;
    }
#endif

    MeshData buildTerrainMesh(const ESM::Land& land, const std::unordered_map<std::uint16_t, std::string>& landTextures)
    {
        ESM::LandRecordData data;
        land.loadData(ESM::Land::DATA_VHGT | ESM::Land::DATA_VNML | ESM::Land::DATA_VCLR | ESM::Land::DATA_VTEX, data);

        MeshData out;
        constexpr unsigned N = ESM::LandRecordData::sLandSize; // 65
        constexpr unsigned T = ESM::LandRecordData::sLandTextureSize; // 16
        const float cellSize = static_cast<float>(ESM::Land::REAL_SIZE);
        const float step = cellSize / static_cast<float>(N - 1);
        const float originX = static_cast<float>(land.mX) * cellSize;
        const float originY = static_cast<float>(land.mY) * cellSize;

        const bool hasHeights = (data.mDataLoaded & ESM::Land::DATA_VHGT) != 0;
        const bool hasNormals = (data.mDataLoaded & ESM::Land::DATA_VNML) != 0;
        const bool hasColors = (data.mDataLoaded & ESM::Land::DATA_VCLR) != 0;
        const bool hasTextures = (data.mDataLoaded & ESM::Land::DATA_VTEX) != 0;

        out.mVertices.resize(N * N);
        for (unsigned y = 0; y < N; ++y)
        {
            for (unsigned x = 0; x < N; ++x)
            {
                const unsigned idx = y * N + x;
                const float h = hasHeights ? data.mHeights[idx] : static_cast<float>(ESM::Land::DEFAULT_HEIGHT);
                const osg::Vec3f pos(originX + static_cast<float>(x) * step, originY + static_cast<float>(y) * step, h);
                expandBounds(out.mBoundsMin, out.mBoundsMax, out.mHasBounds, pos);

                Vertex& v = out.mVertices[idx];
                v.mPos[0] = pos.x();
                v.mPos[1] = pos.y();
                v.mPos[2] = pos.z();

                if (hasNormals)
                {
                    osg::Vec3f n(static_cast<float>(data.mNormals[idx * 3 + 0]),
                        static_cast<float>(data.mNormals[idx * 3 + 1]), static_cast<float>(data.mNormals[idx * 3 + 2]));
                    if (n.length() > 1e-4f)
                        n.normalize();
                    else
                        n = osg::Vec3f(0.f, 0.f, 1.f);
                    v.mNormal[0] = n.x();
                    v.mNormal[1] = n.y();
                    v.mNormal[2] = n.z();
                }
                else
                {
                    v.mNormal[0] = 0.f;
                    v.mNormal[1] = 0.f;
                    v.mNormal[2] = 1.f;
                }

                v.mUV[0] = static_cast<float>(x) / 4.0f;
                v.mUV[1] = static_cast<float>(y) / 4.0f;

                if (hasColors)
                {
                    v.mColor[0] = static_cast<float>(data.mColours[idx * 3 + 0]) / 255.0f;
                    v.mColor[1] = static_cast<float>(data.mColours[idx * 3 + 1]) / 255.0f;
                    v.mColor[2] = static_cast<float>(data.mColours[idx * 3 + 2]) / 255.0f;
                    v.mColor[3] = 1.0f;
                }
                else
                {
                    v.mColor[0] = 1.0f;
                    v.mColor[1] = 1.0f;
                    v.mColor[2] = 1.0f;
                    v.mColor[3] = 1.0f;
                }
            }
        }

        // Group terrain quads by VTEX land-texture ID so each texture becomes a bindless submesh
        std::unordered_map<std::uint16_t, std::vector<std::uint32_t>> bucketIndices;
        for (unsigned y = 0; y < N - 1; ++y)
        {
            for (unsigned x = 0; x < N - 1; ++x)
            {
                std::uint16_t vtexId = 0;
                if (hasTextures)
                {
                    const unsigned tx = std::min(x / 4u, T - 1u);
                    const unsigned ty = std::min(y / 4u, T - 1u);
                    vtexId = data.mTextures[ty * T + tx];
                }

                auto& inds = bucketIndices[vtexId];
                const std::uint32_t i00 = y * N + x;
                const std::uint32_t i10 = y * N + (x + 1);
                const std::uint32_t i01 = (y + 1) * N + x;
                const std::uint32_t i11 = (y + 1) * N + (x + 1);
                inds.push_back(i00);
                inds.push_back(i10);
                inds.push_back(i11);
                inds.push_back(i00);
                inds.push_back(i11);
                inds.push_back(i01);
            }
        }

        out.mIndices.reserve((N - 1) * (N - 1) * 6);
        const osg::Vec3f boundsCenter = (out.mBoundsMin + out.mBoundsMax) * 0.5f;
        const float boundsRadius = std::max(((out.mBoundsMax - out.mBoundsMin) * 0.5f).length(), 1.0f);

        for (auto& [vtexId, inds] : bucketIndices)
        {
            if (inds.empty())
                continue;
            const std::uint32_t firstIdx = static_cast<std::uint32_t>(out.mIndices.size());
            out.mIndices.insert(out.mIndices.end(), inds.begin(), inds.end());

            MaterialDesc mat;
            if (vtexId > 0)
            {
                if (auto it = landTextures.find(static_cast<std::uint16_t>(vtexId - 1)); it != landTextures.end())
                    mat.mDiffuseTexture = it->second;
            }
            if (mat.mDiffuseTexture.empty())
            {
                mat.mBaseColor[0] = 0.56f;
                mat.mBaseColor[1] = 0.62f;
                mat.mBaseColor[2] = 0.44f;
            }

            const std::uint32_t matIdx = static_cast<std::uint32_t>(out.mMaterials.size());
            out.mMaterials.push_back(std::move(mat));

            Submesh sub;
            sub.mFirstIndex = firstIdx;
            sub.mIndexCount = static_cast<std::uint32_t>(inds.size());
            sub.mMaterialIndex = matIdx;
            sub.mBoundsCenter = boundsCenter;
            sub.mBoundsRadius = boundsRadius;
            out.mSubmeshes.push_back(sub);
            ++out.mSubmeshCount;
        }
        return out;
    }

    MeshData buildWaterPlaneMesh(float centerX, float centerY, float waterHeight, float size)
    {
        MeshData out;
        const float half = size * 0.5f;
        const float xs[2] = { centerX - half, centerX + half };
        const float ys[2] = { centerY - half, centerY + half };

        for (int j = 0; j < 2; ++j)
        {
            for (int i = 0; i < 2; ++i)
            {
                Vertex v{};
                v.mPos[0] = xs[i];
                v.mPos[1] = ys[j];
                v.mPos[2] = waterHeight;
                v.mNormal[0] = 0.f;
                v.mNormal[1] = 0.f;
                v.mNormal[2] = 1.f;
                v.mUV[0] = static_cast<float>(i);
                v.mUV[1] = static_cast<float>(j);
                v.mColor[0] = 1.f;
                v.mColor[1] = 1.f;
                v.mColor[2] = 1.f;
                v.mColor[3] = 1.f;
                out.mVertices.push_back(v);
                expandBounds(out.mBoundsMin, out.mBoundsMax, out.mHasBounds, osg::Vec3f(xs[i], ys[j], waterHeight));
            }
        }
        out.mIndices = { 0, 1, 3, 0, 3, 2 };

        MaterialDesc mat;
        mat.mIsWater = true;
        mat.mIsTransparent = true;
        out.mMaterials.push_back(mat);

        Submesh sub;
        sub.mFirstIndex = 0;
        sub.mIndexCount = 6;
        sub.mMaterialIndex = 0;
        sub.mBoundsCenter = osg::Vec3f(centerX, centerY, waterHeight);
        sub.mBoundsRadius = half * 1.5f;
        out.mSubmeshes.push_back(sub);
        out.mSubmeshCount = 1;
        return out;
    }
}
