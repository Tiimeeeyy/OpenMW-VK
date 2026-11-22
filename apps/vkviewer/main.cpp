#include <components/esm/position.hpp>
#include <components/esm3/cellref.hpp>
#include <components/esm3/esmreader.hpp>
#include <components/esm3/loadacti.hpp>
#include <components/esm3/loadcell.hpp>
#include <components/esm3/loadcont.hpp>
#include <components/esm3/loaddoor.hpp>
#include <components/esm3/loadgmst.hpp>
#include <components/esm3/loadland.hpp>
#include <components/esm3/loadligh.hpp>
#include <components/esm3/loadltex.hpp>
#include <components/esm3/loadstat.hpp>
#include <components/esm3/readerscache.hpp>
#include <components/esmloader/esmdata.hpp>
#include <components/esmloader/lessbyid.hpp>
#include <components/esmloader/load.hpp>
#include <components/esmloader/record.hpp>
#include <components/files/collections.hpp>
#include <components/files/configurationmanager.hpp>
#include <components/files/constrainedfilestream.hpp>
#include <components/files/multidircollection.hpp>
#include <components/misc/convert.hpp>
#include <components/misc/resourcehelpers.hpp>
#include <components/misc/strings/algorithm.hpp>
#include <components/nif/niffile.hpp>
#include <components/toutf8/toutf8.hpp>
#include <components/vfs/manager.hpp>
#include <components/vfs/pathutil.hpp>
#include <components/vfs/recursivedirectoryiterator.hpp>
#include <components/vfs/registerarchives.hpp>
#include <components/vkrender/mesh.hpp>
#include <components/vkrender/renderer.hpp>
#include <components/vkrender/scene.hpp>
#include <components/vkrender/texture.hpp>

#include <SDL.h>
#include <SDL_vulkan.h>

#include <boost/program_options.hpp>

#include <osg/Matrixf>
#include <osg/Vec3f>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace
{
    namespace bpo = boost::program_options;

    osg::Matrixf makeVulkanPerspective(float fovyRadians, float aspect, float zNear, float zFar)
    {
        const float f = 1.0f / std::tan(fovyRadians * 0.5f);
        osg::Matrixf m;
        m.makeIdentity();
        m(0, 0) = f / aspect;
        m(1, 1) = -f;
        m(2, 2) = zFar / (zNear - zFar);
        m(2, 3) = -1.0f;
        m(3, 2) = (zFar * zNear) / (zNear - zFar);
        m(3, 3) = 0.0f;
        return m;
    }

    osg::Matrixf makeVulkanOrtho(float left, float right, float bottom, float top, float zNear, float zFar)
    {
        osg::Matrixf m;
        m.makeIdentity();
        m(0, 0) = 2.0f / (right - left);
        m(1, 1) = -2.0f / (top - bottom);
        m(2, 2) = 1.0f / (zNear - zFar);
        m(3, 0) = -(right + left) / (right - left);
        m(3, 1) = (top + bottom) / (top - bottom);
        m(3, 2) = zNear / (zNear - zFar);
        return m;
    }

    osg::Matrixf makeLookAt(const osg::Vec3f& eye, const osg::Vec3f& center, const osg::Vec3f& up)
    {
        osg::Vec3f f = center - eye;
        f.normalize();
        osg::Vec3f s = f ^ up;
        if (s.length() < 1e-4f)
            s = f ^ osg::Vec3f(1.f, 0.f, 0.f);
        s.normalize();
        osg::Vec3f u = s ^ f;

        osg::Matrixf m;
        m.makeIdentity();
        m(0, 0) = s.x();
        m(1, 0) = s.y();
        m(2, 0) = s.z();
        m(3, 0) = -(s * eye);

        m(0, 1) = u.x();
        m(1, 1) = u.y();
        m(2, 1) = u.z();
        m(3, 1) = -(u * eye);

        m(0, 2) = -f.x();
        m(1, 2) = -f.y();
        m(2, 2) = -f.z();
        m(3, 2) = f * eye;
        return m;
    }

    struct Options
    {
        std::string mMeshPath;
        std::string mCellName;
        bool mListMeshes{ false };
        bool mListCells{ false };
        bool mValidation{ false };
        bool mNoShadows{ false };
        int mMsaaSamples{ 4 };
        int mMaxFrames{ 0 };
        Files::PathContainer mDataDirs;
        std::vector<std::string> mArchives;
        std::vector<std::string> mContentFiles;
    };

    bool parseArgs(int argc, char** argv, Options& out)
    {
        bpo::options_description desc("OpenMW Experimental Raw Vulkan Renderer (Bindless + Compute Cull + MDI)");
        auto add = desc.add_options();
        add("help,h", "Show this help message");
        add("mesh,m", bpo::value<std::string>()->default_value(""),
            "Single-mesh mode: VFS path (e.g. meshes/...) or filesystem path (.nif, .dae, .gltf, .glb, .obj).");
        add("cell,c", bpo::value<std::string>()->default_value(""),
            "ESM cell mode: cell name or exterior grid 'x,y' (e.g. '0,0').");
        add("list", bpo::bool_switch(&out.mListMeshes), "List available meshes in mounted VFS and exit");
        add("list-cells", bpo::bool_switch(&out.mListCells), "List available cells in content files and exit");
        add("validation", bpo::bool_switch(&out.mValidation), "Enable Vulkan validation layers");
        add("no-shadows", bpo::bool_switch(&out.mNoShadows), "Disable directional shadow map pass");
        add("msaa", bpo::value<int>(&out.mMsaaSamples)->default_value(4), "MSAA sample count (2 or 4)");
        add("frames", bpo::value<int>(&out.mMaxFrames)->default_value(0),
            "Exit cleanly after rendering N frames (0 = run until closed)");
        add("data",
            bpo::value<Files::MaybeQuotedPathContainer>()
                ->default_value(Files::MaybeQuotedPathContainer(), "")
                ->multitoken()
                ->composing(),
            "Data directories");
        add("data-local",
            bpo::value<Files::MaybeQuotedPathContainer::value_type>()->default_value(
                Files::MaybeQuotedPathContainer::value_type(), ""),
            "Local data directory");
        add("fallback-archive",
            bpo::value<std::vector<std::string>>()
                ->default_value(std::vector<std::string>(), "")
                ->multitoken()
                ->composing(),
            "BSA archives");
        add("content",
            bpo::value<std::vector<std::string>>()
                ->default_value(std::vector<std::string>(), "")
                ->multitoken()
                ->composing(),
            "Content files (.omwgame, .esm, .esp)");

        Files::ConfigurationManager::addCommonOptions(desc);

        bpo::variables_map variables;
        bpo::store(bpo::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), variables);
        bpo::notify(variables);

        if (variables.count("help"))
        {
            std::cout << desc << std::endl;
            return false;
        }

        Files::ConfigurationManager cfgMgr(true);
        cfgMgr.processPaths(variables, std::filesystem::current_path());
        cfgMgr.readConfiguration(variables, desc, true);

        out.mMeshPath = variables["mesh"].as<std::string>();
        out.mCellName = variables["cell"].as<std::string>();
        out.mDataDirs = Files::asPathContainer(variables["data"].as<Files::MaybeQuotedPathContainer>());
        const auto local = variables["data-local"].as<Files::MaybeQuotedPathContainer::value_type>();
        if (!local.empty())
            out.mDataDirs.push_back(local);
        cfgMgr.filterOutNonExistingPaths(out.mDataDirs);
        out.mArchives = variables["fallback-archive"].as<std::vector<std::string>>();
        out.mContentFiles = variables["content"].as<std::vector<std::string>>();
        return true;
    }

    bool isSupportedMesh(std::string_view path)
    {
        if (Misc::StringUtils::ciEndsWith(path, ".nif"))
            return true;
#ifdef OPENMW_VKRENDER_USE_ASSIMP
        if (Misc::StringUtils::ciEndsWith(path, ".dae") || Misc::StringUtils::ciEndsWith(path, ".gltf")
            || Misc::StringUtils::ciEndsWith(path, ".glb") || Misc::StringUtils::ciEndsWith(path, ".obj"))
            return true;
#endif
        return false;
    }

    std::vector<std::byte> readStreamBytes(std::istream& stream)
    {
        stream.seekg(0, std::ios::end);
        const std::streamoff size = stream.tellg();
        stream.seekg(0, std::ios::beg);
        if (size > 0)
        {
            std::vector<std::byte> bytes(static_cast<std::size_t>(size));
            stream.read(reinterpret_cast<char*>(bytes.data()), size);
            bytes.resize(static_cast<std::size_t>(stream.gcount()));
            return bytes;
        }
        std::vector<std::byte> bytes;
        char chunk[4096];
        while (stream.read(chunk, sizeof(chunk)) || stream.gcount() > 0)
        {
            const auto n = static_cast<std::size_t>(stream.gcount());
            const std::size_t prev = bytes.size();
            bytes.resize(prev + n);
            std::memcpy(bytes.data() + prev, chunk, n);
        }
        return bytes;
    }

    struct AssetCache
    {
        VkRender::Renderer& mRenderer;
        const VFS::Manager& mVfs;
        std::unordered_map<std::string, VkRender::TextureId> mTextures;
        std::unordered_map<std::string, VkRender::MeshId> mMeshes;

        VkRender::TextureId resolveTexture(const std::string& rawPath, const std::filesystem::path& meshDir = {})
        {
            if (rawPath.empty())
                return 0;

            std::string normalized = rawPath;
            std::replace(normalized.begin(), normalized.end(), '\\', '/');
            const std::string key = Misc::StringUtils::lowerCase(normalized);
            if (auto it = mTextures.find(key); it != mTextures.end())
                return it->second;

            auto tryLoadStream = [&](Files::IStreamPtr stream, std::string_view ext) -> VkRender::TextureId {
                if (!stream)
                    return 0;
                try
                {
                    const auto bytes = readStreamBytes(*stream);
                    const VkRender::DecodedImage img = VkRender::decodeImage(bytes, ext);
                    if (img.valid())
                    {
                        const VkRender::TextureId id = mRenderer.uploadTexture(img);
                        mTextures[key] = id;
                        return id;
                    }
                }
                catch (...)
                {
                }
                return 0;
            };

            // 1. Try direct/relative filesystem path if loading a loose file
            if (!meshDir.empty())
            {
                const std::filesystem::path rel = meshDir / normalized;
                if (std::filesystem::is_regular_file(rel))
                    if (auto id = tryLoadStream(Files::openConstrainedFileStream(rel), rel.extension().string());
                        id != 0)
                        return id;
                const std::filesystem::path fileOnly = meshDir / std::filesystem::path(normalized).filename();
                if (std::filesystem::is_regular_file(fileOnly))
                    if (auto id
                        = tryLoadStream(Files::openConstrainedFileStream(fileOnly), fileOnly.extension().string());
                        id != 0)
                        return id;
            }

            // 2. Try VFS candidates (exact, textures/<path>, .dds/.png/.tga swaps, basename in textures/)
            const std::string filename = std::filesystem::path(normalized).filename().string();
            const std::vector<std::string> baseCandidates = {
                normalized,
                "textures/" + normalized,
                "textures/" + filename,
            };

            for (const std::string& cand : baseCandidates)
            {
                VFS::Path::Normalized p(cand);
                if (mVfs.exists(p))
                {
                    const std::string ext = std::filesystem::path(cand).extension().string();
                    if (auto id = tryLoadStream(mVfs.get(p), ext); id != 0)
                        return id;
                }
                // Try .dds / .png fallback (common in OpenMW/Morrowind when NIF references .tga/.bmp)
                const std::filesystem::path cp(cand);
                for (const char* altExt : { ".dds", ".png", ".tga" })
                {
                    std::filesystem::path alt = cp;
                    alt.replace_extension(altExt);
                    VFS::Path::Normalized ap(alt.generic_string());
                    if (mVfs.exists(ap))
                    {
                        if (auto id = tryLoadStream(mVfs.get(ap), altExt); id != 0)
                            return id;
                    }
                }
            }

            mTextures[key] = 0;
            return 0;
        }

        std::vector<VkRender::MaterialId> registerMaterials(
            const VkRender::MeshData& data, const std::filesystem::path& meshDir = {})
        {
            std::vector<VkRender::MaterialId> ids;
            ids.reserve(data.mMaterials.size());
            for (const auto& mat : data.mMaterials)
            {
                VkRender::GpuMaterial gpu{};
                std::memcpy(gpu.mBaseColor, mat.mBaseColor, sizeof(gpu.mBaseColor));
                gpu.mAlbedoTexId = resolveTexture(mat.mDiffuseTexture, meshDir);
                if (!mat.mDiffuseTexture.empty())
                {
                    std::filesystem::path normPath(mat.mDiffuseTexture);
                    const std::string stem = normPath.stem().string() + "_n";
                    normPath.replace_filename(stem + normPath.extension().string());
                    gpu.mNormalTexId = resolveTexture(normPath.generic_string(), meshDir);
                }
                gpu.mFlags = VkRender::MaterialFlag_None;
                if (mat.mAlphaTest)
                    gpu.mFlags |= VkRender::MaterialFlag_AlphaTest;
                if (mat.mIsWater)
                    gpu.mFlags |= VkRender::MaterialFlag_Water;
                if (mat.mIsTransparent)
                    gpu.mFlags |= VkRender::MaterialFlag_Transparent;
                gpu.mAlphaCutoff = mat.mAlphaCutoff;
                ids.push_back(mRenderer.addMaterial(gpu));
            }
            return ids;
        }

        VkRender::MeshId loadAndCacheMesh(
            const std::string& requestedMesh, VkRender::MeshData* outRawData = nullptr, std::string* outName = nullptr)
        {
            const std::string cacheKey = Misc::StringUtils::lowerCase(requestedMesh);
            if (outRawData == nullptr)
            {
                if (auto it = mMeshes.find(cacheKey); it != mMeshes.end())
                    return it->second;
            }

            Files::IStreamPtr stream;
            std::string extension;
            std::string loadedName;
            std::filesystem::path meshDir;

            const std::filesystem::path fsPath(requestedMesh);
            if (!requestedMesh.empty() && std::filesystem::is_regular_file(fsPath))
            {
                loadedName = fsPath.string();
                extension = fsPath.extension().string();
                meshDir = fsPath.parent_path();
                stream = Files::openConstrainedFileStream(fsPath);
            }
            else
            {
                VFS::Path::Normalized vfsPath(requestedMesh);
                if (requestedMesh.empty())
                {
                    for (const auto& entry : mVfs.getRecursiveDirectoryIterator())
                    {
                        if (isSupportedMesh(entry.value()))
                        {
                            vfsPath = entry;
                            break;
                        }
                    }
                    if (vfsPath.empty())
                        return VkRender::sInvalidHandle;
                }
                else if (!mVfs.exists(vfsPath))
                {
                    VFS::Path::Normalized prefixed("meshes/" + requestedMesh);
                    if (mVfs.exists(prefixed))
                        vfsPath = prefixed;
                    else
                        return VkRender::sInvalidHandle;
                }

                loadedName = std::string(vfsPath.value());
                extension = std::filesystem::path(loadedName).extension().string();
                stream = mVfs.get(vfsPath);
            }

            VkRender::MeshData data;
            if (Misc::StringUtils::ciEqual(extension, ".nif"))
            {
                Nif::NIFFile file{ VFS::Path::Normalized(loadedName) };
                Nif::Reader reader(file, nullptr);
                reader.parse(std::move(stream));
                data = VkRender::extractNifMesh(Nif::FileView(file));
            }
            else
            {
#ifdef OPENMW_VKRENDER_USE_ASSIMP
                const auto bytes = readStreamBytes(*stream);
                data = VkRender::extractAssimpMesh(bytes, extension);
#else
                return VkRender::sInvalidHandle;
#endif
            }

            if (data.mVertices.empty() || data.mIndices.empty())
            {
                mMeshes[cacheKey] = VkRender::sInvalidHandle;
                return VkRender::sInvalidHandle;
            }

            const auto matIds = registerMaterials(data, meshDir);
            const VkRender::MeshId id = mRenderer.addMesh(data, matIds);
            mMeshes[cacheKey] = id;
            if (outName != nullptr)
                *outName = loadedName;
            if (outRawData != nullptr)
                *outRawData = std::move(data);
            return id;
        }
    };

    struct ExtraEsmRecords
    {
        std::unordered_map<std::uint16_t, std::string> mLandTextures;
        std::unordered_map<ESM::RefId, ESM::Light> mLights;
    };

    ExtraEsmRecords loadExtraEsmRecords(
        const std::vector<std::string>& esmFiles, const Files::Collections& collections, ToUTF8::Utf8Encoder* encoder)
    {
        ExtraEsmRecords extra;
        for (const std::string& filename : esmFiles)
        {
            const std::filesystem::path fullPath = collections.getPath(filename);
            if (!std::filesystem::is_regular_file(fullPath))
                continue;
            ESM::ESMReader reader;
            reader.setEncoder(encoder);
            reader.open(fullPath);
            while (reader.hasMoreRecs())
            {
                const ESM::NAME recName = reader.getRecName();
                reader.getRecHeader();
                if (recName.toInt() == ESM::REC_LTEX)
                {
                    ESM::LandTexture ltex;
                    bool deleted = false;
                    ltex.load(reader, deleted);
                    if (!deleted && !ltex.mTexture.empty())
                        extra.mLandTextures[static_cast<std::uint16_t>(ltex.mIndex)]
                            = std::string(ltex.mTexture.getNormalized().value());
                }
                else if (recName.toInt() == ESM::REC_LIGH)
                {
                    ESM::Light light;
                    bool deleted = false;
                    light.load(reader, deleted);
                    if (!deleted)
                        extra.mLights[light.mId] = std::move(light);
                }
                else
                {
                    reader.skipRecord();
                }
            }
        }
        return extra;
    }

    struct DynamicLightState
    {
        VkRender::LightId mId{ VkRender::sInvalidHandle };
        VkRender::GpuPointLight mBase{};
        std::int32_t mFlags{ 0 };
        float mPhase{ 0.f };
    };

    struct CellRefEntry
    {
        ESM::RecNameInts mType;
        ESM::RefNum mRefNum;
        ESM::RefId mRefId;
        float mScale;
        ESM::Position mPos;
    };

    ESM::RecNameInts findRecordType(const EsmLoader::EsmData& esmData, const ESM::RefId& refId)
    {
        const auto it
            = std::lower_bound(esmData.mRefIdTypes.begin(), esmData.mRefIdTypes.end(), refId, EsmLoader::LessById{});
        if (it == esmData.mRefIdTypes.end() || it->mId != refId)
            return {};
        return it->mType;
    }

    const ESM::Cell* findCell(const EsmLoader::EsmData& esmData, const std::string& cellQuery)
    {
        if (esmData.mCells.empty())
            return nullptr;
        if (cellQuery.empty())
        {
            // Prefer (0,0) exterior cell if present, else first cell with refs
            for (const auto& c : esmData.mCells)
                if (c.isExterior() && c.getGridX() == 0 && c.getGridY() == 0)
                    return &c;
            return &esmData.mCells.front();
        }

        const auto comma = cellQuery.find(',');
        if (comma != std::string::npos)
        {
            try
            {
                const int gx = std::stoi(cellQuery.substr(0, comma));
                const int gy = std::stoi(cellQuery.substr(comma + 1));
                for (const auto& c : esmData.mCells)
                    if (c.isExterior() && c.getGridX() == gx && c.getGridY() == gy)
                        return &c;
            }
            catch (...)
            {
            }
        }

        for (const auto& c : esmData.mCells)
            if (Misc::StringUtils::ciEqual(c.mName, cellQuery))
                return &c;
        return nullptr;
    }
}

int main(int argc, char** argv)
{
    try
    {
        Options opts;
        if (!parseArgs(argc, argv, opts))
            return 0;

        VFS::Manager vfs;
        Files::Collections collections(opts.mDataDirs);
        if (!opts.mDataDirs.empty())
            VFS::registerArchives(&vfs, collections, opts.mArchives, true, nullptr);

        if (opts.mListMeshes)
        {
            std::size_t count = 0;
            for (const auto& entry : vfs.getRecursiveDirectoryIterator())
            {
                if (isSupportedMesh(entry.value()))
                {
                    std::cout << entry.value() << '\n';
                    ++count;
                }
            }
            std::cout << "Total supported meshes: " << count << std::endl;
            return 0;
        }

        const bool cellMode
            = opts.mListCells || !opts.mCellName.empty() || (opts.mMeshPath.empty() && !opts.mContentFiles.empty());

        std::unique_ptr<ESM::ReadersCache> readers;
        std::optional<EsmLoader::EsmData> esmData;
        ExtraEsmRecords extraRecords;
        if (cellMode && !opts.mContentFiles.empty())
        {
            readers = std::make_unique<ESM::ReadersCache>();
            ToUTF8::Utf8Encoder encoder(ToUTF8::WINDOWS_1252);
            EsmLoader::Query query;
            query.mLoadActivators = true;
            query.mLoadCells = true;
            query.mLoadContainers = true;
            query.mLoadDoors = true;
            query.mLoadLands = true;
            query.mLoadStatics = true;

            std::vector<std::string> esmFiles;
            for (const auto& f : opts.mContentFiles)
            {
                if (!Misc::StringUtils::ciEndsWith(f, ".omwscripts"))
                    esmFiles.push_back(f);
            }
            if (!esmFiles.empty())
            {
                esmData.emplace(EsmLoader::loadEsmData(query, esmFiles, collections, *readers, &encoder));
                extraRecords = loadExtraEsmRecords(esmFiles, collections, &encoder);
                std::cout << "[vkviewer] Loaded " << extraRecords.mLandTextures.size() << " LTEX land textures and "
                          << extraRecords.mLights.size() << " LIGH light records" << std::endl;
            }
        }

        if (opts.mListCells)
        {
            if (esmData.has_value())
            {
                for (const auto& cell : esmData->mCells)
                {
                    if (cell.isExterior())
                        std::cout << "Exterior (" << cell.getGridX() << "," << cell.getGridY() << ") " << cell.mName
                                  << '\n';
                    else
                        std::cout << "Interior \"" << cell.mName << "\"\n";
                }
                std::cout << "Total cells: " << esmData->mCells.size() << " | Land chunks: " << esmData->mLands.size()
                          << std::endl;
            }
            else
            {
                std::cout << "No content files loaded. Pass --content <file.omwgame> or configure openmw.cfg."
                          << std::endl;
            }
            return 0;
        }

        if (SDL_Init(SDL_INIT_VIDEO) != 0)
            throw std::runtime_error(std::string("SDL_Init failed: ") + SDL_GetError());

        SDL_Window* window = SDL_CreateWindow("OpenMW Experimental Vulkan Renderer", SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED, 1280, 720, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_SHOWN);
        if (window == nullptr)
        {
            SDL_Quit();
            throw std::runtime_error(std::string("SDL_CreateWindow failed: ") + SDL_GetError());
        }

        {
            VkRender::RendererConfig cfg;
            cfg.mEnableValidation = opts.mValidation;
            cfg.mShaderDir = OPENMW_VK_SHADER_DIR;
            cfg.mMsaaSamples = static_cast<std::uint32_t>(std::max(opts.mMsaaSamples, 2));

            VkRender::Renderer renderer(window, cfg);
            AssetCache cache{ renderer, vfs, {}, {} };

            bool flyCamera = false;
            bool isExteriorScene = false;
            std::vector<DynamicLightState> dynamicLights;
            osg::Vec3f sceneCenter(0.f, 0.f, 0.f);
            float sceneRadius = 100.f;
            osg::Vec3f camPos(0.f, -200.f, 100.f);
            float camYaw = 1.57f;
            float camPitch = -0.2f;

            if (cellMode && esmData.has_value() && !esmData->mCells.empty())
            {
                const ESM::Cell* targetCell = findCell(*esmData, opts.mCellName);
                if (targetCell == nullptr)
                    throw std::runtime_error("Cell not found: " + opts.mCellName);

                flyCamera = true;
                std::cout << "[vkviewer] Loading cell: "
                          << (targetCell->isExterior()
                                     ? ("Exterior (" + std::to_string(targetCell->getGridX()) + ","
                                           + std::to_string(targetCell->getGridY()) + ") " + targetCell->mName)
                                     : targetCell->mName)
                          << std::endl;

                std::vector<const ESM::Cell*> cellsToLoad = { targetCell };
                if (targetCell->isExterior())
                {
                    // Load 3x3 neighborhood around target exterior cell for a full horizon view
                    for (const auto& c : esmData->mCells)
                    {
                        if (&c == targetCell || !c.isExterior())
                            continue;
                        if (std::abs(c.getGridX() - targetCell->getGridX()) <= 1
                            && std::abs(c.getGridY() - targetCell->getGridY()) <= 1)
                            cellsToLoad.push_back(&c);
                    }
                }

                osg::Vec3f boundsMin(1e9f, 1e9f, 1e9f);
                osg::Vec3f boundsMax(-1e9f, -1e9f, -1e9f);
                bool hasBounds = false;
                auto expand = [&](const osg::Vec3f& p) {
                    if (!hasBounds)
                    {
                        boundsMin = p;
                        boundsMax = p;
                        hasBounds = true;
                    }
                    else
                    {
                        boundsMin.x() = std::min(boundsMin.x(), p.x());
                        boundsMin.y() = std::min(boundsMin.y(), p.y());
                        boundsMin.z() = std::min(boundsMin.z(), p.z());
                        boundsMax.x() = std::max(boundsMax.x(), p.x());
                        boundsMax.y() = std::max(boundsMax.y(), p.y());
                        boundsMax.z() = std::max(boundsMax.z(), p.z());
                    }
                };

                // Terrain chunks for exterior cells (multi-texture VTEX/LTEX splatting)
                if (targetCell->isExterior())
                {
                    isExteriorScene = true;
                    for (const auto& land : esmData->mLands)
                    {
                        if (std::abs(land.mX - targetCell->getGridX()) <= 1
                            && std::abs(land.mY - targetCell->getGridY()) <= 1)
                        {
                            VkRender::MeshData terrain = VkRender::buildTerrainMesh(land, extraRecords.mLandTextures);
                            const auto matIds = cache.registerMaterials(terrain);
                            const VkRender::MeshId mid = renderer.addMesh(terrain, matIds);
                            if (mid != VkRender::sInvalidHandle)
                            {
                                renderer.addInstance(mid, osg::Matrixf::identity());
                                if (terrain.mHasBounds)
                                {
                                    expand(terrain.mBoundsMin);
                                    expand(terrain.mBoundsMax);
                                }
                            }
                        }
                    }

                    const float cx = (static_cast<float>(targetCell->getGridX()) + 0.5f) * ESM::Land::REAL_SIZE;
                    const float cy = (static_cast<float>(targetCell->getGridY()) + 0.5f) * ESM::Land::REAL_SIZE;
                    VkRender::MeshData water = VkRender::buildWaterPlaneMesh(cx, cy, 0.0f, ESM::Land::REAL_SIZE * 4.0f);
                    const auto matIds = cache.registerMaterials(water);
                    const VkRender::MeshId waterId = renderer.addMesh(water, matIds);
                    if (waterId != VkRender::sInvalidHandle)
                        renderer.addInstance(waterId, osg::Matrixf::identity());
                }

                // Cell references (statics, activators, containers, doors, AND ESM::Light sources)
                for (const ESM::Cell* cell : cellsToLoad)
                {
                    std::vector<EsmLoader::Record<CellRefEntry>> rawRefs;
                    for (std::size_t i = 0; i < cell->mContextList.size(); ++i)
                    {
                        auto reader = readers->get(static_cast<std::size_t>(cell->mContextList[i].index));
                        cell->restore(*reader, static_cast<int>(i));
                        ESM::CellRef ref;
                        bool deleted = false;
                        while (ESM::Cell::getNextRef(*reader, ref, deleted))
                        {
                            ESM::RecNameInts type = findRecordType(*esmData, ref.mRefID);
                            if (type == ESM::RecNameInts{} && extraRecords.mLights.contains(ref.mRefID))
                                type = ESM::REC_LIGH;
                            if (type == ESM::RecNameInts{})
                                continue;
                            rawRefs.emplace_back(
                                deleted, type, ref.mRefNum, std::move(ref.mRefID), ref.mScale, ref.mPos);
                        }
                    }
                    const auto getKey
                        = [](const EsmLoader::Record<CellRefEntry>& r) -> ESM::RefNum { return r.mValue.mRefNum; };
                    const std::vector<CellRefEntry> refs = EsmLoader::prepareRecords(rawRefs, getKey);

                    for (const auto& ref : refs)
                    {
                        if (ref.mType == ESM::REC_LIGH)
                        {
                            auto it = extraRecords.mLights.find(ref.mRefId);
                            if (it == extraRecords.mLights.end())
                                continue;
                            const ESM::Light& l = it->second;
                            if ((l.mData.mFlags & ESM::Light::OffDefault) == 0 && l.mData.mRadius > 0)
                            {
                                VkRender::GpuPointLight pl{};
                                pl.mPosRadius[0] = ref.mPos.pos[0];
                                pl.mPosRadius[1] = ref.mPos.pos[1];
                                pl.mPosRadius[2] = ref.mPos.pos[2];
                                pl.mPosRadius[3] = static_cast<float>(l.mData.mRadius);
                                const float sign = (l.mData.mFlags & ESM::Light::Negative) ? -1.0f : 1.0f;
                                pl.mColorIntensity[0] = static_cast<float>(l.mData.mColor & 0xFFu) / 255.0f;
                                pl.mColorIntensity[1] = static_cast<float>((l.mData.mColor >> 8u) & 0xFFu) / 255.0f;
                                pl.mColorIntensity[2] = static_cast<float>((l.mData.mColor >> 16u) & 0xFFu) / 255.0f;
                                pl.mColorIntensity[3] = 1.6f * sign;

                                DynamicLightState dls;
                                dls.mId = renderer.addPointLight(pl);
                                dls.mBase = pl;
                                dls.mFlags = l.mData.mFlags;
                                dls.mPhase = static_cast<float>(dynamicLights.size()) * 1.37f;
                                dynamicLights.push_back(dls);
                            }
                            if (!l.mModel.empty())
                            {
                                VFS::Path::Normalized model
                                    = Misc::ResourceHelpers::correctMeshPath(l.mModel.getNormalized());
                                const VkRender::MeshId mid = cache.loadAndCacheMesh(std::string(model.value()));
                                if (mid != VkRender::sInvalidHandle)
                                {
                                    const float s = (ref.mScale > 0.f) ? ref.mScale : 1.0f;
                                    const osg::Matrixf world = osg::Matrixf::scale(s, s, s)
                                        * osg::Matrixf::rotate(Misc::Convert::makeOsgQuat(ref.mPos))
                                        * osg::Matrixf::translate(ref.mPos.asVec3());
                                    renderer.addInstance(mid, world);
                                }
                            }
                            expand(ref.mPos.asVec3());
                            continue;
                        }

                        VFS::Path::Normalized model(EsmLoader::getModel(*esmData, ref.mRefId, ref.mType));
                        if (model.empty())
                            continue;
                        model = Misc::ResourceHelpers::correctMeshPath(model);
                        const VkRender::MeshId mid = cache.loadAndCacheMesh(std::string(model.value()));
                        if (mid == VkRender::sInvalidHandle)
                            continue;

                        const float s = (ref.mScale > 0.f) ? ref.mScale : 1.0f;
                        const osg::Matrixf world = osg::Matrixf::scale(s, s, s)
                            * osg::Matrixf::rotate(Misc::Convert::makeOsgQuat(ref.mPos))
                            * osg::Matrixf::translate(ref.mPos.asVec3());
                        renderer.addInstance(mid, world);
                        expand(ref.mPos.asVec3());
                    }
                }

                if (hasBounds)
                {
                    sceneCenter = (boundsMin + boundsMax) * 0.5f;
                    sceneRadius = std::max(((boundsMax - boundsMin) * 0.5f).length(), 500.0f);
                }
                else if (targetCell->isExterior())
                {
                    sceneCenter = osg::Vec3f((static_cast<float>(targetCell->getGridX()) + 0.5f) * ESM::Land::REAL_SIZE,
                        (static_cast<float>(targetCell->getGridY()) + 0.5f) * ESM::Land::REAL_SIZE, 200.f);
                    sceneRadius = 4096.f;
                }
                camPos = sceneCenter + osg::Vec3f(0.f, -std::min(sceneRadius * 0.6f, 1200.f), 350.f);

                if (dynamicLights.empty())
                {
                    VkRender::GpuPointLight demoLight{};
                    demoLight.mPosRadius[0] = sceneCenter.x();
                    demoLight.mPosRadius[1] = sceneCenter.y();
                    demoLight.mPosRadius[2] = sceneCenter.z() + 180.f;
                    demoLight.mPosRadius[3] = 1200.f;
                    demoLight.mColorIntensity[0] = 1.0f;
                    demoLight.mColorIntensity[1] = 0.78f;
                    demoLight.mColorIntensity[2] = 0.45f;
                    demoLight.mColorIntensity[3] = 1.4f;
                    renderer.addPointLight(demoLight);
                }
            }
            else
            {
                VkRender::MeshData rawMesh;
                std::string loadedName;
                const VkRender::MeshId mid = cache.loadAndCacheMesh(opts.mMeshPath, &rawMesh, &loadedName);
                if (mid == VkRender::sInvalidHandle)
                    throw std::runtime_error(
                        "Failed to load mesh: " + (opts.mMeshPath.empty() ? std::string("<auto>") : opts.mMeshPath));

                std::cout << "[vkviewer] Loaded mesh: " << loadedName << std::endl;
                renderer.addInstance(mid, osg::Matrixf::identity());
                sceneCenter = (rawMesh.mBoundsMin + rawMesh.mBoundsMax) * 0.5f;
                sceneRadius = std::max(((rawMesh.mBoundsMax - rawMesh.mBoundsMin) * 0.5f).length(), 1.0f);
            }

            renderer.commitScene();
            const VkRender::SceneStats stats = renderer.getSceneStats();
            std::cout << "[vkviewer] Scene committed: " << stats.mMeshCount << " meshes, " << stats.mTextureCount
                      << " bindless textures, " << stats.mInstanceCount << " MDI instances, " << stats.mTriangleCount
                      << " unique triangles, " << stats.mPointLightCount << " point lights" << std::endl;

            float orbitYaw = 0.7f;
            float orbitPitch = 0.35f;
            float orbitDist = sceneRadius * 2.6f;
            bool dragging = false;
            bool enableShadows = !opts.mNoShadows;
            bool enableFrustumCull = true;
            bool enableGTAO = true;
            bool enableBloom = true;
            bool enableToneMap = true;
            bool running = true;
            int renderedFrames = 0;

            std::uint64_t prevTicks = SDL_GetPerformanceCounter();
            const double perfFreq = static_cast<double>(SDL_GetPerformanceFrequency());
            float elapsed = 0.f;

            while (running)
            {
                const std::uint64_t nowTicks = SDL_GetPerformanceCounter();
                const float dt = static_cast<float>(static_cast<double>(nowTicks - prevTicks) / perfFreq);
                prevTicks = nowTicks;
                elapsed += dt;

                SDL_Event ev;
                while (SDL_PollEvent(&ev))
                {
                    if (ev.type == SDL_QUIT)
                        running = false;
                    else if (ev.type == SDL_WINDOWEVENT
                        && (ev.window.event == SDL_WINDOWEVENT_RESIZED
                            || ev.window.event == SDL_WINDOWEVENT_SIZE_CHANGED))
                        renderer.notifyResized();
                    else if (ev.type == SDL_KEYDOWN)
                    {
                        if (ev.key.keysym.sym == SDLK_ESCAPE || ev.key.keysym.sym == SDLK_q)
                            running = false;
                        else if (ev.key.keysym.sym == SDLK_f)
                        {
                            enableFrustumCull = !enableFrustumCull;
                            std::cout << "[vkviewer] Compute frustum culling: " << (enableFrustumCull ? "ON" : "OFF")
                                      << std::endl;
                        }
                        else if (ev.key.keysym.sym == SDLK_b)
                        {
                            enableShadows = !enableShadows;
                            std::cout << "[vkviewer] 3-Cascade Shadows: " << (enableShadows ? "ON" : "OFF")
                                      << std::endl;
                        }
                        else if (ev.key.keysym.sym == SDLK_g)
                        {
                            enableGTAO = !enableGTAO;
                            std::cout << "[vkviewer] Compute GTAO: " << (enableGTAO ? "ON" : "OFF") << std::endl;
                        }
                        else if (ev.key.keysym.sym == SDLK_l)
                        {
                            enableBloom = !enableBloom;
                            std::cout << "[vkviewer] Bloom: " << (enableBloom ? "ON" : "OFF") << std::endl;
                        }
                        else if (ev.key.keysym.sym == SDLK_t)
                        {
                            enableToneMap = !enableToneMap;
                            std::cout << "[vkviewer] ACES Tone Mapping: " << (enableToneMap ? "ON" : "OFF")
                                      << std::endl;
                        }
                        else if (ev.key.keysym.sym == SDLK_r)
                        {
                            orbitYaw = 0.7f;
                            orbitPitch = 0.35f;
                            orbitDist = sceneRadius * 2.6f;
                            camPos = sceneCenter + osg::Vec3f(0.f, -std::min(sceneRadius * 0.6f, 1200.f), 350.f);
                            camYaw = 1.57f;
                            camPitch = -0.2f;
                        }
                    }
                    else if (ev.type == SDL_MOUSEBUTTONDOWN && ev.button.button == SDL_BUTTON_LEFT)
                        dragging = true;
                    else if (ev.type == SDL_MOUSEBUTTONUP && ev.button.button == SDL_BUTTON_LEFT)
                        dragging = false;
                    else if (ev.type == SDL_MOUSEMOTION && dragging)
                    {
                        if (flyCamera)
                        {
                            camYaw -= static_cast<float>(ev.motion.xrel) * 0.004f;
                            camPitch = std::clamp(camPitch - static_cast<float>(ev.motion.yrel) * 0.004f, -1.5f, 1.5f);
                        }
                        else
                        {
                            orbitYaw -= static_cast<float>(ev.motion.xrel) * 0.008f;
                            orbitPitch
                                = std::clamp(orbitPitch + static_cast<float>(ev.motion.yrel) * 0.008f, -1.45f, 1.45f);
                        }
                    }
                    else if (ev.type == SDL_MOUSEWHEEL && !flyCamera)
                    {
                        orbitDist *= (ev.wheel.y > 0) ? 0.9f : 1.1f;
                        orbitDist = std::clamp(orbitDist, sceneRadius * 0.25f, sceneRadius * 20.0f);
                    }
                }

                osg::Vec3f eye;
                osg::Vec3f target;
                if (flyCamera)
                {
                    const osg::Vec3f forward(std::cos(camPitch) * std::cos(camYaw),
                        std::cos(camPitch) * std::sin(camYaw), std::sin(camPitch));
                    osg::Vec3f right = forward ^ osg::Vec3f(0.f, 0.f, 1.f);
                    right.normalize();

                    const Uint8* keys = SDL_GetKeyboardState(nullptr);
                    const float speed = (keys[SDL_SCANCODE_LSHIFT] ? 2400.f : 650.f) * dt;
                    if (keys[SDL_SCANCODE_W])
                        camPos += forward * speed;
                    if (keys[SDL_SCANCODE_S])
                        camPos -= forward * speed;
                    if (keys[SDL_SCANCODE_D])
                        camPos += right * speed;
                    if (keys[SDL_SCANCODE_A])
                        camPos -= right * speed;
                    if (keys[SDL_SCANCODE_SPACE] || keys[SDL_SCANCODE_E])
                        camPos.z() += speed;
                    if (keys[SDL_SCANCODE_LCTRL] || keys[SDL_SCANCODE_C])
                        camPos.z() -= speed;

                    eye = camPos;
                    target = camPos + forward;
                }
                else
                {
                    eye = osg::Vec3f(sceneCenter.x() + orbitDist * std::cos(orbitPitch) * std::cos(orbitYaw),
                        sceneCenter.y() + orbitDist * std::cos(orbitPitch) * std::sin(orbitYaw),
                        sceneCenter.z() + orbitDist * std::sin(orbitPitch));
                    target = sceneCenter;
                }

                int dw = 1;
                int dh = 1;
                SDL_Vulkan_GetDrawableSize(window, &dw, &dh);
                const float aspect = (dh > 0) ? (static_cast<float>(dw) / static_cast<float>(dh)) : 1.0f;

                const float zNear = flyCamera ? 4.0f : std::max(sceneRadius * 0.01f, 0.1f);
                const float zFar = flyCamera ? 48000.0f : sceneRadius * 50.0f;
                const osg::Matrixf viewMat = makeLookAt(eye, target, osg::Vec3f(0.f, 0.f, 1.f));
                const osg::Matrixf projMat = makeVulkanPerspective(0.785398f, aspect, zNear, zFar);

                osg::Vec3f sunDir(0.45f, -0.55f, 0.70f);
                sunDir.normalize();
                const osg::Vec3f shadowFocus = flyCamera ? eye : sceneCenter;
                const std::array<float, VkRender::sShadowCascadeCount> cascadeExtents = flyCamera
                    ? std::array<float, 3>{ 650.0f, 2400.0f, 9500.0f }
                    : std::array<float, 3>{ std::max(sceneRadius * 1.4f, 10.f), std::max(sceneRadius * 3.0f, 25.f),
                          std::max(sceneRadius * 6.0f, 60.f) };

                // Animate flickering / pulsing ESM::Light sources each frame via persistent-mapped SSBOs
                for (const auto& dl : dynamicLights)
                {
                    if ((dl.mFlags
                            & (ESM::Light::Flicker | ESM::Light::FlickerSlow | ESM::Light::Pulse | ESM::Light::PulseSlow
                                | ESM::Light::Fire))
                        == 0)
                        continue;
                    VkRender::GpuPointLight pl = dl.mBase;
                    const float freq = (dl.mFlags & (ESM::Light::Flicker | ESM::Light::Fire)) ? 9.0f : 2.5f;
                    const float mod = 0.85f + 0.15f * std::sin(elapsed * freq + dl.mPhase)
                        + 0.08f * std::sin(elapsed * freq * 2.3f + dl.mPhase * 1.7f);
                    pl.mColorIntensity[3] *= mod;
                    renderer.updatePointLight(dl.mId, pl);
                }

                VkRender::FrameView fv;
                fv.mCascadeSplits = osg::Vec3f(cascadeExtents[0], cascadeExtents[1], cascadeExtents[2]);
                for (std::uint32_t c = 0; c < VkRender::sShadowCascadeCount; ++c)
                {
                    const float ext = cascadeExtents[c];
                    osg::Matrixf lightView
                        = makeLookAt(shadowFocus + sunDir * (ext * 1.5f), shadowFocus, osg::Vec3f(0.f, 0.f, 1.f));
                    const float texelSize = (ext * 2.0f) / 2048.0f;
                    const osg::Vec3f originLS = osg::Vec3f(0.f, 0.f, 0.f) * lightView;
                    lightView(3, 0) += std::floor(originLS.x() / texelSize) * texelSize - originLS.x();
                    lightView(3, 1) += std::floor(originLS.y() / texelSize) * texelSize - originLS.y();
                    const osg::Matrixf lightProj = makeVulkanOrtho(-ext, ext, -ext, ext, 1.0f, ext * 4.0f);
                    fv.mCascadeViewProj[c] = lightView * lightProj;
                }
                fv.mView = viewMat;
                fv.mProj = projMat;
                fv.mViewProj = viewMat * projMat;
                fv.mCameraPos = eye;
                fv.mTime = elapsed;
                fv.mSunDir = sunDir;
                fv.mFogColor = isExteriorScene ? osg::Vec3f(0.55f, 0.65f, 0.78f) : osg::Vec3f(0.08f, 0.10f, 0.14f);
                fv.mFogStart = isExteriorScene ? 10000.f : 0.f;
                fv.mFogEnd = isExteriorScene ? 28000.f : 0.f;
                fv.mMaxDrawDistance = flyCamera ? 32000.f : 0.f;
                fv.mEnableShadows = enableShadows;
                fv.mEnableFrustumCull = enableFrustumCull;
                fv.mEnableSky = isExteriorScene;
                fv.mEnableGTAO = enableGTAO;
                fv.mEnableBloom = enableBloom;
                fv.mEnableToneMap = enableToneMap;

                renderer.drawFrame(fv);
                ++renderedFrames;
                if (opts.mMaxFrames > 0 && renderedFrames >= opts.mMaxFrames)
                    running = false;
            }
        }

        SDL_DestroyWindow(window);
        SDL_Quit();
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[vkviewer] Error: " << e.what() << std::endl;
        return 1;
    }
}
