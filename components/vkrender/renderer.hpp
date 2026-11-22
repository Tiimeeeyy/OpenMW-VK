#ifndef OPENMW_COMPONENTS_VKRENDER_RENDERER_HPP
#define OPENMW_COMPONENTS_VKRENDER_RENDERER_HPP

#include "mesh.hpp"
#include "scene.hpp"
#include "texture.hpp"

#include <osg/Matrixf>
#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

struct SDL_Window;

namespace VkRender
{
    struct RendererConfig
    {
        bool mEnableValidation{ false };
        std::filesystem::path mShaderDir;
        std::uint32_t mShadowMapSize{ 2048 };
        std::uint32_t mMsaaSamples{ 4 }; // 1, 2, or 4
    };

    struct SceneStats
    {
        std::size_t mMeshCount{ 0 };
        std::size_t mTextureCount{ 0 };
        std::size_t mMaterialCount{ 0 };
        std::size_t mInstanceCount{ 0 };
        std::size_t mVertexCount{ 0 };
        std::size_t mTriangleCount{ 0 };
        std::size_t mPointLightCount{ 0 };
    };

    class Renderer
    {
    public:
        Renderer(SDL_Window* window, const RendererConfig& config);
        ~Renderer();

        Renderer(const Renderer&) = delete;
        Renderer& operator=(const Renderer&) = delete;

        TextureId uploadTexture(const DecodedImage& image);
        MaterialId addMaterial(const GpuMaterial& material);
        MeshId addMesh(const MeshData& mesh, const std::vector<MaterialId>& resolvedMaterials);

        InstanceId addInstance(
            MeshId meshId, const osg::Matrixf& worldMatrix, std::uint32_t boneOffset = sInvalidHandle);
        std::uint32_t getMeshSubmeshCount(MeshId meshId) const;
        void updateInstanceTransform(InstanceId firstId, MeshId meshId, const osg::Matrixf& worldMatrix);
        void removeInstance(InstanceId firstId, std::uint32_t submeshCount);

        LightId addPointLight(const GpuPointLight& light);
        void updatePointLight(LightId id, const GpuPointLight& light);
        void removePointLight(LightId id);

        std::uint32_t allocateBonePalette(std::uint32_t boneCount);
        void updateBonePalette(std::uint32_t boneOffset, std::span<const osg::Matrixf> bones);

        void commitScene();
        void clearScene();
        SceneStats getSceneStats() const;

        void notifyResized() { mFramebufferResized = true; }
        void drawFrame(const FrameView& view);

    private:
        static constexpr std::uint32_t sMaxFramesInFlight = 2;
        static constexpr std::uint32_t sMaxBindlessTextures = 4096;
        static constexpr std::uint32_t sInitialInstanceCap = 4096;
        static constexpr std::uint32_t sInitialMaterialCap = 1024;
        static constexpr std::uint32_t sInitialLightCap = 512;
        static constexpr std::uint32_t sMaxBones = 4096;

        struct QueueFamilyIndices
        {
            std::uint32_t mGraphics{ UINT32_MAX };
            std::uint32_t mPresent{ UINT32_MAX };
            bool isComplete() const { return mGraphics != UINT32_MAX && mPresent != UINT32_MAX; }
        };

        struct AllocatedBuffer
        {
            VkBuffer mBuffer{ VK_NULL_HANDLE };
            VkDeviceMemory mMemory{ VK_NULL_HANDLE };
            VkDeviceSize mSize{ 0 };
            void* mMapped{ nullptr };
        };

        struct AllocatedImage
        {
            VkImage mImage{ VK_NULL_HANDLE };
            VkDeviceMemory mMemory{ VK_NULL_HANDLE };
            VkImageView mView{ VK_NULL_HANDLE };
        };

        struct StagedSubmesh
        {
            std::uint32_t mFirstIndex{ 0 };
            std::uint32_t mIndexCount{ 0 };
            std::int32_t mVertexOffset{ 0 };
            MaterialId mMaterialId{ 0 };
            bool mTransparent{ false };
            osg::Vec3f mBoundsCenter{ 0.f, 0.f, 0.f };
            float mBoundsRadius{ 1.f };
        };

        struct StagedMesh
        {
            std::vector<StagedSubmesh> mSubmeshes;
        };

        void createInstance(bool enableValidation);
        void createSurface();
        void pickPhysicalDevice();
        void createLogicalDevice();
        void createCommandPool();
        void createSwapchain();
        void createOffscreenTargets();
        void createShadowResources();
        void createRenderPasses();
        void createFramebuffers();
        void createDescriptors();
        void createPipelines();
        void createCommandBuffers();
        void createSyncObjects();
        void createDefaultResources();

        void cleanupSwapchain();
        void recreateSwapchain();
        void recreateTileBuffers();
        void ensureDynamicBufferCapacities(std::uint32_t frameIdx);
        void syncDynamicBuffers(std::uint32_t frameIdx);
        void updateFrameDescriptors(std::uint32_t frameIdx);
        void updatePostDescriptors();

        QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) const;
        std::uint32_t findMemoryType(std::uint32_t typeFilter, VkMemoryPropertyFlags properties) const;
        VkFormat findDepthFormat() const;
        VkSampleCountFlagBits chooseSampleCount(std::uint32_t requested) const;

        AllocatedBuffer createBuffer(
            VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, bool persistentMap = false);
        void destroyBuffer(AllocatedBuffer& buf);
        void uploadToDeviceBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage, AllocatedBuffer& out);

        AllocatedImage createImage2D(std::uint32_t w, std::uint32_t h, VkFormat format, VkImageUsageFlags usage,
            VkImageAspectFlags aspect, VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT,
            std::uint32_t mipLevels = 1);
        void destroyImage(AllocatedImage& img);

        VkCommandBuffer beginSingleTimeCommands();
        void endSingleTimeCommands(VkCommandBuffer cmd);

        void transitionImageLayout(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
            std::uint32_t mipLevels = 1, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT,
            std::uint32_t layerCount = 1);

        VkShaderModule createShaderModule(const std::filesystem::path& spvPath) const;

        SDL_Window* mWindow{ nullptr };
        std::filesystem::path mShaderDir;
        std::uint32_t mShadowMapSize{ 2048 };
        std::uint32_t mRequestedMsaa{ 4 };
        VkSampleCountFlagBits mMsaaSamples{ VK_SAMPLE_COUNT_4_BIT };
        bool mValidationEnabled{ false };
        bool mSupportsAnisotropy{ false };

        VkInstance mInstance{ VK_NULL_HANDLE };
        VkDebugUtilsMessengerEXT mDebugMessenger{ VK_NULL_HANDLE };
        VkSurfaceKHR mSurface{ VK_NULL_HANDLE };
        VkPhysicalDevice mPhysicalDevice{ VK_NULL_HANDLE };
        VkDevice mDevice{ VK_NULL_HANDLE };
        QueueFamilyIndices mQueueFamilies;
        VkQueue mGraphicsQueue{ VK_NULL_HANDLE };
        VkQueue mPresentQueue{ VK_NULL_HANDLE };

        VkCommandPool mCommandPool{ VK_NULL_HANDLE };
        std::array<VkCommandBuffer, sMaxFramesInFlight> mCommandBuffers{};

        VkSwapchainKHR mSwapchain{ VK_NULL_HANDLE };
        VkFormat mSwapchainImageFormat{ VK_FORMAT_UNDEFINED };
        VkExtent2D mSwapchainExtent{ 0, 0 };
        std::vector<VkImage> mSwapchainImages;
        std::vector<VkImageView> mSwapchainImageViews;

        // Offscreen float scene targets + MSAA + resolved depth + GTAO + Bloom
        VkFormat mColorFormat{ VK_FORMAT_R16G16B16A16_SFLOAT };
        VkFormat mDepthFormat{ VK_FORMAT_UNDEFINED };
        AllocatedImage mMsaaColorImage;
        AllocatedImage mMsaaDepthImage;
        AllocatedImage mResolvedColorImage;
        AllocatedImage mResolvedDepthImage; // R32_SFLOAT sampled by GTAO + depth-aware water
        AllocatedImage mAoImage; // Half-res R8_UNORM
        AllocatedImage mBloomA; // Half-res R16G16B16A16_SFLOAT
        AllocatedImage mBloomB; // Half-res R16G16B16A16_SFLOAT

        // 3-Cascade Shadow Map (2D_ARRAY)
        VkImage mShadowImage{ VK_NULL_HANDLE };
        VkDeviceMemory mShadowMemory{ VK_NULL_HANDLE };
        VkImageView mShadowArrayView{ VK_NULL_HANDLE };
        std::array<VkImageView, sShadowCascadeCount> mShadowLayerViews{};
        VkSampler mShadowSampler{ VK_NULL_HANDLE };
        VkRenderPass mShadowRenderPass{ VK_NULL_HANDLE };
        std::array<VkFramebuffer, sShadowCascadeCount> mShadowFramebuffers{};

        // Render passes & framebuffers
        VkRenderPass mOpaqueRenderPass{ VK_NULL_HANDLE };
        VkRenderPass mTransparentRenderPass{ VK_NULL_HANDLE };
        VkRenderPass mPostRenderPass{ VK_NULL_HANDLE };
        VkFramebuffer mOpaqueFramebuffer{ VK_NULL_HANDLE };
        VkFramebuffer mTransparentFramebuffer{ VK_NULL_HANDLE };
        std::vector<VkFramebuffer> mSwapchainFramebuffers;

        // Samplers & Descriptors
        VkSampler mLinearSampler{ VK_NULL_HANDLE };
        VkSampler mClampSampler{ VK_NULL_HANDLE };
        VkDescriptorSetLayout mGlobalSetLayout{ VK_NULL_HANDLE };
        VkDescriptorSetLayout mCullSetLayout{ VK_NULL_HANDLE };
        VkDescriptorSetLayout mCompute2ImgSetLayout{ VK_NULL_HANDLE }; // 1 sampler + 1 storage image
        VkDescriptorSetLayout mPostSetLayout{ VK_NULL_HANDLE }; // sceneColor + bloom + ao
        VkDescriptorPool mDescriptorPool{ VK_NULL_HANDLE };
        std::array<VkDescriptorSet, sMaxFramesInFlight> mGlobalSets{};
        std::array<VkDescriptorSet, sMaxFramesInFlight> mCullSets{};
        VkDescriptorSet mDepthResolveSet{ VK_NULL_HANDLE };
        VkDescriptorSet mGtaoSet{ VK_NULL_HANDLE };
        VkDescriptorSet mBloomDownSet{ VK_NULL_HANDLE };
        VkDescriptorSet mBloomBlurSet{ VK_NULL_HANDLE };
        VkDescriptorSet mPostSet{ VK_NULL_HANDLE };

        // Pipelines
        VkPipelineLayout mGraphicsPipelineLayout{ VK_NULL_HANDLE };
        VkPipeline mOpaquePipeline{ VK_NULL_HANDLE };
        VkPipeline mTransparentPipeline{ VK_NULL_HANDLE };
        VkPipeline mSkyPipeline{ VK_NULL_HANDLE };
        VkPipeline mShadowPipeline{ VK_NULL_HANDLE };

        VkPipelineLayout mCullPipelineLayout{ VK_NULL_HANDLE };
        VkPipeline mCullPipeline{ VK_NULL_HANDLE };
        VkPipeline mLightCullPipeline{ VK_NULL_HANDLE };

        VkPipelineLayout mDepthResolveLayout{ VK_NULL_HANDLE };
        VkPipeline mDepthResolvePipeline{ VK_NULL_HANDLE };

        VkPipelineLayout mGtaoLayout{ VK_NULL_HANDLE };
        VkPipeline mGtaoPipeline{ VK_NULL_HANDLE };

        VkPipelineLayout mBloomLayout{ VK_NULL_HANDLE };
        VkPipeline mBloomPipeline{ VK_NULL_HANDLE };

        VkPipelineLayout mPostLayout{ VK_NULL_HANDLE };
        VkPipeline mPostPipeline{ VK_NULL_HANDLE };

        // Per-frame persistent-mapped dynamic buffers + GPU-driven indirect & tile-light buffers
        std::array<AllocatedBuffer, sMaxFramesInFlight> mFrameUbos{};
        std::array<AllocatedBuffer, sMaxFramesInFlight> mMaterialBuffers{};
        std::array<AllocatedBuffer, sMaxFramesInFlight> mInstanceBuffers{};
        std::array<AllocatedBuffer, sMaxFramesInFlight> mLightBuffers{};
        std::array<AllocatedBuffer, sMaxFramesInFlight> mBoneBuffers{};
        std::array<AllocatedBuffer, sMaxFramesInFlight> mOpaqueIndirectBuffers{};
        std::array<AllocatedBuffer, sMaxFramesInFlight> mOpaqueCountBuffers{};
        std::array<AllocatedBuffer, sMaxFramesInFlight> mTransparentIndirectBuffers{};
        std::array<AllocatedBuffer, sMaxFramesInFlight> mTransparentCountBuffers{};
        std::array<AllocatedBuffer, sMaxFramesInFlight> mTileLightBuffers{};

        std::array<std::uint32_t, sMaxFramesInFlight> mInstanceCapacities{};
        std::array<std::uint32_t, sMaxFramesInFlight> mMaterialCapacities{};
        std::array<std::uint32_t, sMaxFramesInFlight> mLightCapacities{};
        std::uint32_t mTileCountX{ 1 };
        std::uint32_t mTileCountY{ 1 };

        // Mega vertex & index buffers
        AllocatedBuffer mVertexBuffer;
        AllocatedBuffer mIndexBuffer;
        bool mGeometryDirty{ false };

        // Bindless textures
        std::vector<AllocatedImage> mTextures;

        // CPU scene state
        std::vector<Vertex> mStagedVertices;
        std::vector<std::uint32_t> mStagedIndices;
        std::vector<StagedMesh> mStagedMeshes;
        std::vector<GpuMaterial> mStagedMaterials;
        std::vector<GpuInstance> mStagedInstances;
        std::vector<std::uint32_t> mFreeInstanceSlots;
        std::vector<GpuPointLight> mStagedLights;
        std::vector<osg::Matrixf> mStagedBones;
        std::array<bool, sMaxFramesInFlight> mMaterialsDirty{ true, true };
        std::array<bool, sMaxFramesInFlight> mInstancesDirty{ true, true };
        std::array<bool, sMaxFramesInFlight> mLightsDirty{ true, true };
        std::array<bool, sMaxFramesInFlight> mBonesDirty{ true, true };

        // Sync
        std::array<VkSemaphore, sMaxFramesInFlight> mImageAvailableSemaphores{};
        std::vector<VkSemaphore> mRenderFinishedSemaphores;
        std::array<VkFence, sMaxFramesInFlight> mInFlightFences{};
        std::uint32_t mCurrentFrame{ 0 };
        bool mFramebufferResized{ false };
    };
}

#endif
