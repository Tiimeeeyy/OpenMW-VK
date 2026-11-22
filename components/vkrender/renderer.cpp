#include "renderer.hpp"

#include <SDL_vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

namespace VkRender
{
    namespace
    {
        constexpr const char* sValidationLayerName = "VK_LAYER_KHRONOS_validation";

        struct CullPushConstants
        {
            std::uint32_t mInstanceCount;
            std::uint32_t mEnableFrustumCull;
            std::uint32_t mOpaqueOnly;
        };

        struct ShadowPushConstants
        {
            std::uint32_t mCascadeIndex;
        };

        struct DepthResolvePushConstants
        {
            std::uint32_t mWidth;
            std::uint32_t mHeight;
            std::uint32_t mSampleCount;
        };

        struct BloomPushConstants
        {
            std::uint32_t mPassMode;
            float mThreshold;
        };

        void vkCheck(VkResult result, const char* context)
        {
            if (result != VK_SUCCESS)
                throw std::runtime_error(std::string(context) + " failed (VkResult=" + std::to_string(result) + ")");
        }

        VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
            VkDebugUtilsMessageTypeFlagsEXT /*type*/, const VkDebugUtilsMessengerCallbackDataEXT* data,
            void* /*userData*/)
        {
            if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
                std::cerr << "[Vulkan] " << data->pMessage << std::endl;
            return VK_FALSE;
        }

        bool hasValidationLayerSupport()
        {
            std::uint32_t count = 0;
            vkEnumerateInstanceLayerProperties(&count, nullptr);
            std::vector<VkLayerProperties> layers(count);
            vkEnumerateInstanceLayerProperties(&count, layers.data());
            for (const auto& layer : layers)
            {
                if (std::strcmp(layer.layerName, sValidationLayerName) == 0)
                    return true;
            }
            return false;
        }

        std::vector<char> readBinaryFile(const std::filesystem::path& path)
        {
            std::ifstream file(path, std::ios::ate | std::ios::binary);
            if (!file.is_open())
                throw std::runtime_error("Failed to open shader file: " + path.string());
            const std::streamsize size = file.tellg();
            std::vector<char> buffer(static_cast<std::size_t>(size));
            file.seekg(0);
            file.read(buffer.data(), size);
            return buffer;
        }

        float computeMaxScale(const osg::Matrixf& m)
        {
            const float sx = osg::Vec3f(m(0, 0), m(0, 1), m(0, 2)).length();
            const float sy = osg::Vec3f(m(1, 0), m(1, 1), m(1, 2)).length();
            const float sz = osg::Vec3f(m(2, 0), m(2, 1), m(2, 2)).length();
            return std::max({ sx, sy, sz, 1e-4f });
        }
    }

    Renderer::Renderer(SDL_Window* window, const RendererConfig& config)
        : mWindow(window)
        , mShaderDir(config.mShaderDir)
        , mShadowMapSize(config.mShadowMapSize)
        , mRequestedMsaa(config.mMsaaSamples)
    {
        if (mWindow == nullptr)
            throw std::runtime_error("VkRender::Renderer requires a non-null SDL_Window");

        createInstance(config.mEnableValidation);
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createCommandPool();
        createSwapchain();
        createOffscreenTargets();
        createShadowResources();
        createRenderPasses();
        createFramebuffers();
        createDescriptors();
        createPipelines();
        createCommandBuffers();
        createSyncObjects();
        createDefaultResources();
    }

    Renderer::~Renderer()
    {
        if (mDevice != VK_NULL_HANDLE)
            vkDeviceWaitIdle(mDevice);

        cleanupSwapchain();

        for (auto& tex : mTextures)
            destroyImage(tex);
        mTextures.clear();

        destroyBuffer(mVertexBuffer);
        destroyBuffer(mIndexBuffer);

        for (std::uint32_t i = 0; i < sMaxFramesInFlight; ++i)
        {
            destroyBuffer(mFrameUbos[i]);
            destroyBuffer(mMaterialBuffers[i]);
            destroyBuffer(mInstanceBuffers[i]);
            destroyBuffer(mLightBuffers[i]);
            destroyBuffer(mBoneBuffers[i]);
            destroyBuffer(mOpaqueIndirectBuffers[i]);
            destroyBuffer(mOpaqueCountBuffers[i]);
            destroyBuffer(mTransparentIndirectBuffers[i]);
            destroyBuffer(mTransparentCountBuffers[i]);
            destroyBuffer(mTileLightBuffers[i]);
        }

        for (auto& fb : mShadowFramebuffers)
        {
            if (fb != VK_NULL_HANDLE)
                vkDestroyFramebuffer(mDevice, fb, nullptr);
        }
        if (mShadowRenderPass != VK_NULL_HANDLE)
            vkDestroyRenderPass(mDevice, mShadowRenderPass, nullptr);
        if (mShadowSampler != VK_NULL_HANDLE)
            vkDestroySampler(mDevice, mShadowSampler, nullptr);
        for (auto& v : mShadowLayerViews)
        {
            if (v != VK_NULL_HANDLE)
                vkDestroyImageView(mDevice, v, nullptr);
        }
        if (mShadowArrayView != VK_NULL_HANDLE)
            vkDestroyImageView(mDevice, mShadowArrayView, nullptr);
        if (mShadowImage != VK_NULL_HANDLE)
            vkDestroyImage(mDevice, mShadowImage, nullptr);
        if (mShadowMemory != VK_NULL_HANDLE)
            vkFreeMemory(mDevice, mShadowMemory, nullptr);

        if (mClampSampler != VK_NULL_HANDLE)
            vkDestroySampler(mDevice, mClampSampler, nullptr);
        if (mLinearSampler != VK_NULL_HANDLE)
            vkDestroySampler(mDevice, mLinearSampler, nullptr);

        if (mPostPipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(mDevice, mPostPipeline, nullptr);
        if (mPostLayout != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(mDevice, mPostLayout, nullptr);
        if (mBloomPipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(mDevice, mBloomPipeline, nullptr);
        if (mBloomLayout != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(mDevice, mBloomLayout, nullptr);
        if (mGtaoPipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(mDevice, mGtaoPipeline, nullptr);
        if (mGtaoLayout != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(mDevice, mGtaoLayout, nullptr);
        if (mDepthResolvePipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(mDevice, mDepthResolvePipeline, nullptr);
        if (mDepthResolveLayout != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(mDevice, mDepthResolveLayout, nullptr);

        if (mLightCullPipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(mDevice, mLightCullPipeline, nullptr);
        if (mCullPipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(mDevice, mCullPipeline, nullptr);
        if (mCullPipelineLayout != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(mDevice, mCullPipelineLayout, nullptr);
        if (mShadowPipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(mDevice, mShadowPipeline, nullptr);
        if (mSkyPipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(mDevice, mSkyPipeline, nullptr);
        if (mTransparentPipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(mDevice, mTransparentPipeline, nullptr);
        if (mOpaquePipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(mDevice, mOpaquePipeline, nullptr);
        if (mGraphicsPipelineLayout != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(mDevice, mGraphicsPipelineLayout, nullptr);

        if (mPostRenderPass != VK_NULL_HANDLE)
            vkDestroyRenderPass(mDevice, mPostRenderPass, nullptr);
        if (mTransparentRenderPass != VK_NULL_HANDLE)
            vkDestroyRenderPass(mDevice, mTransparentRenderPass, nullptr);
        if (mOpaqueRenderPass != VK_NULL_HANDLE)
            vkDestroyRenderPass(mDevice, mOpaqueRenderPass, nullptr);

        if (mDescriptorPool != VK_NULL_HANDLE)
            vkDestroyDescriptorPool(mDevice, mDescriptorPool, nullptr);
        if (mPostSetLayout != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(mDevice, mPostSetLayout, nullptr);
        if (mCompute2ImgSetLayout != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(mDevice, mCompute2ImgSetLayout, nullptr);
        if (mCullSetLayout != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(mDevice, mCullSetLayout, nullptr);
        if (mGlobalSetLayout != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(mDevice, mGlobalSetLayout, nullptr);

        for (std::uint32_t i = 0; i < sMaxFramesInFlight; ++i)
        {
            if (mImageAvailableSemaphores[i] != VK_NULL_HANDLE)
                vkDestroySemaphore(mDevice, mImageAvailableSemaphores[i], nullptr);
            if (mInFlightFences[i] != VK_NULL_HANDLE)
                vkDestroyFence(mDevice, mInFlightFences[i], nullptr);
        }
        for (VkSemaphore sem : mRenderFinishedSemaphores)
        {
            if (sem != VK_NULL_HANDLE)
                vkDestroySemaphore(mDevice, sem, nullptr);
        }

        if (mCommandPool != VK_NULL_HANDLE)
            vkDestroyCommandPool(mDevice, mCommandPool, nullptr);
        if (mDevice != VK_NULL_HANDLE)
            vkDestroyDevice(mDevice, nullptr);
        if (mSurface != VK_NULL_HANDLE)
            vkDestroySurfaceKHR(mInstance, mSurface, nullptr);

        if (mDebugMessenger != VK_NULL_HANDLE)
        {
            auto destroyFn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(mInstance, "vkDestroyDebugUtilsMessengerEXT"));
            if (destroyFn != nullptr)
                destroyFn(mInstance, mDebugMessenger, nullptr);
        }

        if (mInstance != VK_NULL_HANDLE)
            vkDestroyInstance(mInstance, nullptr);
    }

    void Renderer::createInstance(bool enableValidation)
    {
        mValidationEnabled = enableValidation && hasValidationLayerSupport();
        if (enableValidation && !mValidationEnabled)
            std::cerr << "[Vulkan] Validation layer requested but not installed; continuing without it." << std::endl;

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "OpenMW Experimental Vulkan Renderer";
        appInfo.applicationVersion = VK_MAKE_VERSION(0, 4, 0);
        appInfo.pEngineName = "OpenMW";
        appInfo.engineVersion = VK_MAKE_VERSION(0, 4, 0);
        appInfo.apiVersion = VK_API_VERSION_1_2;

        unsigned int sdlExtCount = 0;
        if (!SDL_Vulkan_GetInstanceExtensions(mWindow, &sdlExtCount, nullptr))
            throw std::runtime_error(std::string("SDL_Vulkan_GetInstanceExtensions failed: ") + SDL_GetError());
        std::vector<const char*> extensions(sdlExtCount);
        if (!SDL_Vulkan_GetInstanceExtensions(mWindow, &sdlExtCount, extensions.data()))
            throw std::runtime_error(std::string("SDL_Vulkan_GetInstanceExtensions failed: ") + SDL_GetError());

        if (mValidationEnabled)
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        std::vector<const char*> layers;
        if (mValidationEnabled)
            layers.push_back(sValidationLayerName);

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();
        createInfo.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
        createInfo.ppEnabledLayerNames = layers.data();

        vkCheck(vkCreateInstance(&createInfo, nullptr, &mInstance), "vkCreateInstance");

        if (mValidationEnabled)
        {
            VkDebugUtilsMessengerCreateInfoEXT dbgInfo{};
            dbgInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            dbgInfo.messageSeverity
                = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            dbgInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            dbgInfo.pfnUserCallback = debugCallback;

            auto createFn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(mInstance, "vkCreateDebugUtilsMessengerEXT"));
            if (createFn != nullptr)
                vkCheck(createFn(mInstance, &dbgInfo, nullptr, &mDebugMessenger), "vkCreateDebugUtilsMessengerEXT");
        }
    }

    void Renderer::createSurface()
    {
        if (!SDL_Vulkan_CreateSurface(mWindow, mInstance, &mSurface))
            throw std::runtime_error(std::string("SDL_Vulkan_CreateSurface failed: ") + SDL_GetError());
    }

    Renderer::QueueFamilyIndices Renderer::findQueueFamilies(VkPhysicalDevice device) const
    {
        QueueFamilyIndices indices;
        std::uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

        for (std::uint32_t i = 0; i < count; ++i)
        {
            if ((families[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))
                == (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))
                indices.mGraphics = i;

            VkBool32 presentSupport = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, mSurface, &presentSupport);
            if (presentSupport == VK_TRUE)
                indices.mPresent = i;

            if (indices.isComplete())
                break;
        }
        return indices;
    }

    VkSampleCountFlagBits Renderer::chooseSampleCount(std::uint32_t requested) const
    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(mPhysicalDevice, &props);
        const VkSampleCountFlags counts
            = props.limits.framebufferColorSampleCounts & props.limits.framebufferDepthSampleCounts;

        if (requested >= 4 && (counts & VK_SAMPLE_COUNT_4_BIT))
            return VK_SAMPLE_COUNT_4_BIT;
        if (counts & VK_SAMPLE_COUNT_2_BIT)
            return VK_SAMPLE_COUNT_2_BIT;
        return VK_SAMPLE_COUNT_4_BIT;
    }

    void Renderer::pickPhysicalDevice()
    {
        std::uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(mInstance, &deviceCount, nullptr);
        if (deviceCount == 0)
            throw std::runtime_error("No Vulkan-capable physical devices found");

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(mInstance, &deviceCount, devices.data());

        int bestScore = -1;
        for (VkPhysicalDevice dev : devices)
        {
            const QueueFamilyIndices qf = findQueueFamilies(dev);
            if (!qf.isComplete())
                continue;

            std::uint32_t extCount = 0;
            vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount, nullptr);
            std::vector<VkExtensionProperties> exts(extCount);
            vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount, exts.data());
            bool hasSwapchain = false;
            for (const auto& ext : exts)
            {
                if (std::strcmp(ext.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0)
                {
                    hasSwapchain = true;
                    break;
                }
            }
            if (!hasSwapchain)
                continue;

            std::uint32_t formatCount = 0;
            std::uint32_t modeCount = 0;
            vkGetPhysicalDeviceSurfaceFormatsKHR(dev, mSurface, &formatCount, nullptr);
            vkGetPhysicalDeviceSurfacePresentModesKHR(dev, mSurface, &modeCount, nullptr);
            if (formatCount == 0 || modeCount == 0)
                continue;

            VkPhysicalDeviceVulkan12Features vk12{};
            vk12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
            VkPhysicalDeviceFeatures2 feats2{};
            feats2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            feats2.pNext = &vk12;
            vkGetPhysicalDeviceFeatures2(dev, &feats2);

            if (!vk12.descriptorIndexing || !vk12.runtimeDescriptorArray || !vk12.descriptorBindingPartiallyBound
                || !vk12.shaderSampledImageArrayNonUniformIndexing || !vk12.drawIndirectCount
                || !feats2.features.multiDrawIndirect)
                continue;

            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(dev, &props);
            const int score = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ? 1000 : 100;
            if (score > bestScore)
            {
                bestScore = score;
                mPhysicalDevice = dev;
                mQueueFamilies = qf;
                mSupportsAnisotropy = (feats2.features.samplerAnisotropy == VK_TRUE);
            }
        }

        if (mPhysicalDevice == VK_NULL_HANDLE)
            throw std::runtime_error(
                "Failed to find a Vulkan 1.2 GPU supporting bindless descriptors + Multi-Draw Indirect");

        mMsaaSamples = chooseSampleCount(mRequestedMsaa);

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(mPhysicalDevice, &props);
        std::cout << "[Vulkan] Selected GPU: " << props.deviceName << " (" << static_cast<int>(mMsaaSamples)
                  << "x MSAA + 3-CSM + PBR + GTAO + Bloom + ACES + Forward+ MDI)" << std::endl;
    }

    void Renderer::createLogicalDevice()
    {
        const std::set<std::uint32_t> uniqueFamilies = { mQueueFamilies.mGraphics, mQueueFamilies.mPresent };
        const float queuePriority = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> queueInfos;
        for (std::uint32_t family : uniqueFamilies)
        {
            VkDeviceQueueCreateInfo qi{};
            qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            qi.queueFamilyIndex = family;
            qi.queueCount = 1;
            qi.pQueuePriorities = &queuePriority;
            queueInfos.push_back(qi);
        }

        VkPhysicalDeviceVulkan12Features vk12{};
        vk12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vk12.descriptorIndexing = VK_TRUE;
        vk12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
        vk12.descriptorBindingPartiallyBound = VK_TRUE;
        vk12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
        vk12.runtimeDescriptorArray = VK_TRUE;
        vk12.drawIndirectCount = VK_TRUE;

        VkPhysicalDeviceFeatures features{};
        features.multiDrawIndirect = VK_TRUE;
        features.samplerAnisotropy = mSupportsAnisotropy ? VK_TRUE : VK_FALSE;
        features.textureCompressionBC = VK_TRUE;

        const char* deviceExts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pNext = &vk12;
        createInfo.queueCreateInfoCount = static_cast<std::uint32_t>(queueInfos.size());
        createInfo.pQueueCreateInfos = queueInfos.data();
        createInfo.pEnabledFeatures = &features;
        createInfo.enabledExtensionCount = 1;
        createInfo.ppEnabledExtensionNames = deviceExts;

        vkCheck(vkCreateDevice(mPhysicalDevice, &createInfo, nullptr, &mDevice), "vkCreateDevice");

        vkGetDeviceQueue(mDevice, mQueueFamilies.mGraphics, 0, &mGraphicsQueue);
        vkGetDeviceQueue(mDevice, mQueueFamilies.mPresent, 0, &mPresentQueue);
    }

    void Renderer::createCommandPool()
    {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = mQueueFamilies.mGraphics;
        vkCheck(vkCreateCommandPool(mDevice, &poolInfo, nullptr, &mCommandPool), "vkCreateCommandPool");
    }

    void Renderer::createSwapchain()
    {
        VkSurfaceCapabilitiesKHR caps{};
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(mPhysicalDevice, mSurface, &caps);

        std::uint32_t formatCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(mPhysicalDevice, mSurface, &formatCount, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(mPhysicalDevice, mSurface, &formatCount, formats.data());

        VkSurfaceFormatKHR surfaceFormat = formats[0];
        for (const auto& fmt : formats)
        {
            if (fmt.format == VK_FORMAT_B8G8R8A8_SRGB && fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                surfaceFormat = fmt;
                break;
            }
        }

        VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
        std::uint32_t modeCount = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(mPhysicalDevice, mSurface, &modeCount, nullptr);
        std::vector<VkPresentModeKHR> modes(modeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(mPhysicalDevice, mSurface, &modeCount, modes.data());
        for (VkPresentModeKHR mode : modes)
        {
            if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
            {
                presentMode = mode;
                break;
            }
        }

        VkExtent2D extent = caps.currentExtent;
        if (extent.width == UINT32_MAX)
        {
            int w = 0;
            int h = 0;
            SDL_Vulkan_GetDrawableSize(mWindow, &w, &h);
            extent.width
                = std::clamp(static_cast<std::uint32_t>(w), caps.minImageExtent.width, caps.maxImageExtent.width);
            extent.height
                = std::clamp(static_cast<std::uint32_t>(h), caps.minImageExtent.height, caps.maxImageExtent.height);
        }

        std::uint32_t imageCount = caps.minImageCount + 1;
        if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
            imageCount = caps.maxImageCount;

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = mSurface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        const std::uint32_t qfIndices[] = { mQueueFamilies.mGraphics, mQueueFamilies.mPresent };
        if (mQueueFamilies.mGraphics != mQueueFamilies.mPresent)
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = qfIndices;
        }
        else
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform = caps.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;

        vkCheck(vkCreateSwapchainKHR(mDevice, &createInfo, nullptr, &mSwapchain), "vkCreateSwapchainKHR");

        mSwapchainImageFormat = surfaceFormat.format;
        mSwapchainExtent = extent;

        vkGetSwapchainImagesKHR(mDevice, mSwapchain, &imageCount, nullptr);
        mSwapchainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(mDevice, mSwapchain, &imageCount, mSwapchainImages.data());

        mSwapchainImageViews.resize(imageCount);
        for (std::size_t i = 0; i < imageCount; ++i)
        {
            VkImageViewCreateInfo viewInfo{};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = mSwapchainImages[i];
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = mSwapchainImageFormat;
            viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            vkCheck(vkCreateImageView(mDevice, &viewInfo, nullptr, &mSwapchainImageViews[i]), "vkCreateImageView");
        }
    }

    VkFormat Renderer::findDepthFormat() const
    {
        const VkFormat candidates[] = {
            VK_FORMAT_D32_SFLOAT,
            VK_FORMAT_D32_SFLOAT_S8_UINT,
            VK_FORMAT_D24_UNORM_S8_UINT,
        };
        for (VkFormat fmt : candidates)
        {
            VkFormatProperties props{};
            vkGetPhysicalDeviceFormatProperties(mPhysicalDevice, fmt, &props);
            if ((props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
                && (props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT))
                return fmt;
        }
        throw std::runtime_error("Failed to find a supported depth format");
    }

    Renderer::AllocatedImage Renderer::createImage2D(std::uint32_t w, std::uint32_t h, VkFormat format,
        VkImageUsageFlags usage, VkImageAspectFlags aspect, VkSampleCountFlagBits samples, std::uint32_t mipLevels)
    {
        AllocatedImage img;
        VkImageCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        info.imageType = VK_IMAGE_TYPE_2D;
        info.extent = { std::max(1u, w), std::max(1u, h), 1 };
        info.mipLevels = mipLevels;
        info.arrayLayers = 1;
        info.format = format;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        info.usage = usage;
        info.samples = samples;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCheck(vkCreateImage(mDevice, &info, nullptr, &img.mImage), "vkCreateImage(2D)");

        VkMemoryRequirements memReq{};
        vkGetImageMemoryRequirements(mDevice, img.mImage, &memReq);
        VkMemoryAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc.allocationSize = memReq.size;
        alloc.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vkCheck(vkAllocateMemory(mDevice, &alloc, nullptr, &img.mMemory), "vkAllocateMemory(2D)");
        vkCheck(vkBindImageMemory(mDevice, img.mImage, img.mMemory, 0), "vkBindImageMemory(2D)");

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = img.mImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange = { aspect, 0, mipLevels, 0, 1 };
        vkCheck(vkCreateImageView(mDevice, &viewInfo, nullptr, &img.mView), "vkCreateImageView(2D)");
        return img;
    }

    void Renderer::destroyImage(AllocatedImage& img)
    {
        if (img.mView != VK_NULL_HANDLE)
        {
            vkDestroyImageView(mDevice, img.mView, nullptr);
            img.mView = VK_NULL_HANDLE;
        }
        if (img.mImage != VK_NULL_HANDLE)
        {
            vkDestroyImage(mDevice, img.mImage, nullptr);
            img.mImage = VK_NULL_HANDLE;
        }
        if (img.mMemory != VK_NULL_HANDLE)
        {
            vkFreeMemory(mDevice, img.mMemory, nullptr);
            img.mMemory = VK_NULL_HANDLE;
        }
    }

    void Renderer::createOffscreenTargets()
    {
        mDepthFormat = findDepthFormat();
        const std::uint32_t w = mSwapchainExtent.width;
        const std::uint32_t h = mSwapchainExtent.height;
        const std::uint32_t hw = std::max(1u, w / 2u);
        const std::uint32_t hh = std::max(1u, h / 2u);

        mMsaaColorImage = createImage2D(
            w, h, mColorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT, mMsaaSamples);
        mMsaaDepthImage = createImage2D(w, h, mDepthFormat,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT,
            mMsaaSamples);

        mResolvedColorImage = createImage2D(w, h, mColorFormat,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        mResolvedDepthImage = createImage2D(w, h, VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

        mAoImage = createImage2D(hw, hh, VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT);
        mBloomA = createImage2D(
            hw, hh, mColorFormat, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        mBloomB = createImage2D(
            hw, hh, mColorFormat, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

        VkCommandBuffer cmd = beginSingleTimeCommands();
        transitionImageLayout(
            cmd, mResolvedDepthImage.mImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        transitionImageLayout(
            cmd, mAoImage.mImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        transitionImageLayout(cmd, mBloomB.mImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        endSingleTimeCommands(cmd);
    }

    void Renderer::createShadowResources()
    {
        VkImageCreateInfo imgInfo{};
        imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType = VK_IMAGE_TYPE_2D;
        imgInfo.extent = { mShadowMapSize, mShadowMapSize, 1 };
        imgInfo.mipLevels = 1;
        imgInfo.arrayLayers = sShadowCascadeCount;
        imgInfo.format = mDepthFormat;
        imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imgInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        vkCheck(vkCreateImage(mDevice, &imgInfo, nullptr, &mShadowImage), "vkCreateImage(shadow)");

        VkMemoryRequirements memReq{};
        vkGetImageMemoryRequirements(mDevice, mShadowImage, &memReq);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vkCheck(vkAllocateMemory(mDevice, &allocInfo, nullptr, &mShadowMemory), "vkAllocateMemory(shadow)");
        vkCheck(vkBindImageMemory(mDevice, mShadowImage, mShadowMemory, 0), "vkBindImageMemory(shadow)");

        VkImageViewCreateInfo arrayViewInfo{};
        arrayViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        arrayViewInfo.image = mShadowImage;
        arrayViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        arrayViewInfo.format = mDepthFormat;
        arrayViewInfo.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, sShadowCascadeCount };
        vkCheck(vkCreateImageView(mDevice, &arrayViewInfo, nullptr, &mShadowArrayView), "vkCreateImageView(shadowArr)");

        for (std::uint32_t c = 0; c < sShadowCascadeCount; ++c)
        {
            VkImageViewCreateInfo layerViewInfo{};
            layerViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            layerViewInfo.image = mShadowImage;
            layerViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            layerViewInfo.format = mDepthFormat;
            layerViewInfo.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, c, 1 };
            vkCheck(vkCreateImageView(mDevice, &layerViewInfo, nullptr, &mShadowLayerViews[c]),
                "vkCreateImageView(shadowLayer)");
        }

        VkSamplerCreateInfo sampInfo{};
        sampInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampInfo.magFilter = VK_FILTER_LINEAR;
        sampInfo.minFilter = VK_FILTER_LINEAR;
        sampInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sampInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sampInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sampInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        sampInfo.compareEnable = VK_TRUE;
        sampInfo.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        vkCheck(vkCreateSampler(mDevice, &sampInfo, nullptr, &mShadowSampler), "vkCreateSampler(shadow)");

        VkCommandBuffer cmd = beginSingleTimeCommands();
        transitionImageLayout(cmd, mShadowImage, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, 1, VK_IMAGE_ASPECT_DEPTH_BIT, sShadowCascadeCount);
        endSingleTimeCommands(cmd);
    }

    void Renderer::createRenderPasses()
    {
        // 1. Shadow depth-only render pass (per cascade layer)
        {
            VkAttachmentDescription depthAtt{};
            depthAtt.format = mDepthFormat;
            depthAtt.samples = VK_SAMPLE_COUNT_1_BIT;
            depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depthAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            depthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            depthAtt.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

            VkAttachmentReference depthRef{ 0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
            VkSubpassDescription subpass{};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.pDepthStencilAttachment = &depthRef;

            std::array<VkSubpassDependency, 2> deps{};
            deps[0] = { VK_SUBPASS_EXTERNAL, 0, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, 0 };
            deps[1] = { 0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT, 0 };

            VkRenderPassCreateInfo rpInfo{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, nullptr, 0, 1, &depthAtt, 1,
                &subpass, static_cast<std::uint32_t>(deps.size()), deps.data() };
            vkCheck(vkCreateRenderPass(mDevice, &rpInfo, nullptr, &mShadowRenderPass), "vkCreateRenderPass(shadow)");
        }

        // 2. Opaque + Sky MSAA pass (stores MSAA color & transitions MSAA depth to READ_ONLY for depth_resolve)
        {
            VkAttachmentDescription colorAtt{};
            colorAtt.format = mColorFormat;
            colorAtt.samples = mMsaaSamples;
            colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            colorAtt.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            VkAttachmentDescription depthAtt{};
            depthAtt.format = mDepthFormat;
            depthAtt.samples = mMsaaSamples;
            depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depthAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            depthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            depthAtt.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

            VkAttachmentReference colorRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
            VkAttachmentReference depthRef{ 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
            VkSubpassDescription subpass{};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &colorRef;
            subpass.pDepthStencilAttachment = &depthRef;

            VkSubpassDependency dep{ 0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT, 0 };

            const std::array<VkAttachmentDescription, 2> atts = { colorAtt, depthAtt };
            VkRenderPassCreateInfo rpInfo{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, nullptr, 0,
                static_cast<std::uint32_t>(atts.size()), atts.data(), 1, &subpass, 1, &dep };
            vkCheck(vkCreateRenderPass(mDevice, &rpInfo, nullptr, &mOpaqueRenderPass), "vkCreateRenderPass(opaque)");
        }

        // 3. Transparent / Water MSAA pass (loads MSAA color+depth, resolves into 1x mResolvedColorImage)
        {
            VkAttachmentDescription colorAtt{};
            colorAtt.format = mColorFormat;
            colorAtt.samples = mMsaaSamples;
            colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAtt.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            colorAtt.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            VkAttachmentDescription depthAtt{};
            depthAtt.format = mDepthFormat;
            depthAtt.samples = mMsaaSamples;
            depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            depthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAtt.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depthAtt.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            VkAttachmentDescription resolveAtt{};
            resolveAtt.format = mColorFormat;
            resolveAtt.samples = VK_SAMPLE_COUNT_1_BIT;
            resolveAtt.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            resolveAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            resolveAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            resolveAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            resolveAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            resolveAtt.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkAttachmentReference colorRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
            VkAttachmentReference depthRef{ 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
            VkAttachmentReference resolveRef{ 2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

            VkSubpassDescription subpass{};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &colorRef;
            subpass.pDepthStencilAttachment = &depthRef;
            subpass.pResolveAttachments = &resolveRef;

            VkSubpassDependency dep{ 0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, 0 };

            const std::array<VkAttachmentDescription, 3> atts = { colorAtt, depthAtt, resolveAtt };
            VkRenderPassCreateInfo rpInfo{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, nullptr, 0,
                static_cast<std::uint32_t>(atts.size()), atts.data(), 1, &subpass, 1, &dep };
            vkCheck(vkCreateRenderPass(mDevice, &rpInfo, nullptr, &mTransparentRenderPass),
                "vkCreateRenderPass(transparent)");
        }

        // 4. Final Post-Process Swapchain Render Pass (ACES + GTAO + Bloom + CAS -> SDR Swapchain)
        {
            VkAttachmentDescription colorAtt{};
            colorAtt.format = mSwapchainImageFormat;
            colorAtt.samples = VK_SAMPLE_COUNT_1_BIT;
            colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            colorAtt.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

            VkAttachmentReference colorRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
            VkSubpassDescription subpass{};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &colorRef;

            VkSubpassDependency dep{ VK_SUBPASS_EXTERNAL, 0, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0 };

            VkRenderPassCreateInfo rpInfo{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, nullptr, 0, 1, &colorAtt, 1,
                &subpass, 1, &dep };
            vkCheck(vkCreateRenderPass(mDevice, &rpInfo, nullptr, &mPostRenderPass), "vkCreateRenderPass(post)");
        }
    }

    void Renderer::createFramebuffers()
    {
        for (std::uint32_t c = 0; c < sShadowCascadeCount; ++c)
        {
            if (mShadowFramebuffers[c] == VK_NULL_HANDLE)
            {
                VkFramebufferCreateInfo fbInfo{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, nullptr, 0,
                    mShadowRenderPass, 1, &mShadowLayerViews[c], mShadowMapSize, mShadowMapSize, 1 };
                vkCheck(vkCreateFramebuffer(mDevice, &fbInfo, nullptr, &mShadowFramebuffers[c]),
                    "vkCreateFramebuffer(shadow)");
            }
        }

        {
            const std::array<VkImageView, 2> atts = { mMsaaColorImage.mView, mMsaaDepthImage.mView };
            VkFramebufferCreateInfo fbInfo{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, nullptr, 0, mOpaqueRenderPass,
                static_cast<std::uint32_t>(atts.size()), atts.data(), mSwapchainExtent.width, mSwapchainExtent.height,
                1 };
            vkCheck(vkCreateFramebuffer(mDevice, &fbInfo, nullptr, &mOpaqueFramebuffer), "vkCreateFramebuffer(opaque)");
        }

        {
            const std::array<VkImageView, 3> atts
                = { mMsaaColorImage.mView, mMsaaDepthImage.mView, mResolvedColorImage.mView };
            VkFramebufferCreateInfo fbInfo{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, nullptr, 0,
                mTransparentRenderPass, static_cast<std::uint32_t>(atts.size()), atts.data(), mSwapchainExtent.width,
                mSwapchainExtent.height, 1 };
            vkCheck(vkCreateFramebuffer(mDevice, &fbInfo, nullptr, &mTransparentFramebuffer),
                "vkCreateFramebuffer(transparent)");
        }

        mSwapchainFramebuffers.resize(mSwapchainImageViews.size());
        for (std::size_t i = 0; i < mSwapchainImageViews.size(); ++i)
        {
            VkFramebufferCreateInfo fbInfo{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, nullptr, 0, mPostRenderPass, 1,
                &mSwapchainImageViews[i], mSwapchainExtent.width, mSwapchainExtent.height, 1 };
            vkCheck(vkCreateFramebuffer(mDevice, &fbInfo, nullptr, &mSwapchainFramebuffers[i]),
                "vkCreateFramebuffer(post)");
        }
    }

    void Renderer::createDescriptors()
    {
        VkSamplerCreateInfo sampInfo{};
        sampInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampInfo.magFilter = VK_FILTER_LINEAR;
        sampInfo.minFilter = VK_FILTER_LINEAR;
        sampInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sampInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampInfo.anisotropyEnable = mSupportsAnisotropy ? VK_TRUE : VK_FALSE;
        sampInfo.maxAnisotropy = mSupportsAnisotropy ? 8.0f : 1.0f;
        sampInfo.maxLod = VK_LOD_CLAMP_NONE;
        vkCheck(vkCreateSampler(mDevice, &sampInfo, nullptr, &mLinearSampler), "vkCreateSampler(linear)");

        VkSamplerCreateInfo clampInfo{};
        clampInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        clampInfo.magFilter = VK_FILTER_LINEAR;
        clampInfo.minFilter = VK_FILTER_LINEAR;
        clampInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        clampInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        clampInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vkCheck(vkCreateSampler(mDevice, &clampInfo, nullptr, &mClampSampler), "vkCreateSampler(clamp)");

        // Set 0: Global scene + bindless texture array + tile light grid + bone palette + resolved scene depth
        const VkShaderStageFlags allStages
            = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
        std::array<VkDescriptorSetLayoutBinding, 10> globalBindings{};
        globalBindings[0] = { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, allStages, nullptr };
        globalBindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        globalBindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, allStages, nullptr };
        globalBindings[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
            VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        globalBindings[4] = { 4, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        globalBindings[5]
            = { 5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, sMaxBindlessTextures, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        globalBindings[6] = { 6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        globalBindings[7] = { 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        globalBindings[8] = { 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr };
        globalBindings[9] = { 9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };

        std::array<VkDescriptorBindingFlags, 10> bindingFlags{};
        bindingFlags[5] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;

        VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO, nullptr,
            static_cast<std::uint32_t>(bindingFlags.size()), bindingFlags.data()
        };

        VkDescriptorSetLayoutCreateInfo globalLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            &flagsInfo, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
            static_cast<std::uint32_t>(globalBindings.size()), globalBindings.data() };
        vkCheck(vkCreateDescriptorSetLayout(mDevice, &globalLayoutInfo, nullptr, &mGlobalSetLayout),
            "vkCreateDescriptorSetLayout(global)");

        // Set 1: Compute cull output
        std::array<VkDescriptorSetLayoutBinding, 5> cullBindings{};
        for (std::uint32_t b = 0; b < 5; ++b)
            cullBindings[b] = { b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        VkDescriptorSetLayoutCreateInfo cullLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0,
            static_cast<std::uint32_t>(cullBindings.size()), cullBindings.data() };
        vkCheck(vkCreateDescriptorSetLayout(mDevice, &cullLayoutInfo, nullptr, &mCullSetLayout),
            "vkCreateDescriptorSetLayout(cull)");

        // Compute 1-sampler + 1-storage-image layout (depth resolve, GTAO, bloom)
        std::array<VkDescriptorSetLayoutBinding, 2> comp2Bindings{};
        comp2Bindings[0] = { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        comp2Bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        VkDescriptorSetLayoutCreateInfo comp2LayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr,
            0, static_cast<std::uint32_t>(comp2Bindings.size()), comp2Bindings.data() };
        vkCheck(vkCreateDescriptorSetLayout(mDevice, &comp2LayoutInfo, nullptr, &mCompute2ImgSetLayout),
            "vkCreateDescriptorSetLayout(comp2)");

        // Post-process set (sceneColor + bloom + ao)
        std::array<VkDescriptorSetLayoutBinding, 3> postBindings{};
        for (std::uint32_t b = 0; b < 3; ++b)
            postBindings[b]
                = { b, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        VkDescriptorSetLayoutCreateInfo postLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0,
            static_cast<std::uint32_t>(postBindings.size()), postBindings.data() };
        vkCheck(vkCreateDescriptorSetLayout(mDevice, &postLayoutInfo, nullptr, &mPostSetLayout),
            "vkCreateDescriptorSetLayout(post)");

        std::array<VkDescriptorPoolSize, 6> poolSizes{};
        poolSizes[0] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, sMaxFramesInFlight };
        poolSizes[1] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, sMaxFramesInFlight * 10 };
        poolSizes[2] = { VK_DESCRIPTOR_TYPE_SAMPLER, sMaxFramesInFlight };
        poolSizes[3] = { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, sMaxFramesInFlight * sMaxBindlessTextures };
        poolSizes[4] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sMaxFramesInFlight * 2 + 12 };
        poolSizes[5] = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8 };

        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr,
            VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT, sMaxFramesInFlight * 2 + 8,
            static_cast<std::uint32_t>(poolSizes.size()), poolSizes.data() };
        vkCheck(vkCreateDescriptorPool(mDevice, &poolInfo, nullptr, &mDescriptorPool), "vkCreateDescriptorPool");

        auto allocSet = [&](VkDescriptorSetLayout layout) {
            VkDescriptorSet set = VK_NULL_HANDLE;
            VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, mDescriptorPool, 1,
                &layout };
            vkCheck(vkAllocateDescriptorSets(mDevice, &ai, &set), "vkAllocateDescriptorSets");
            return set;
        };

        for (std::uint32_t i = 0; i < sMaxFramesInFlight; ++i)
        {
            mGlobalSets[i] = allocSet(mGlobalSetLayout);
            mCullSets[i] = allocSet(mCullSetLayout);
        }
        mDepthResolveSet = allocSet(mCompute2ImgSetLayout);
        mGtaoSet = allocSet(mCompute2ImgSetLayout);
        mBloomDownSet = allocSet(mCompute2ImgSetLayout);
        mBloomBlurSet = allocSet(mCompute2ImgSetLayout);
        mPostSet = allocSet(mPostSetLayout);

        const VkMemoryPropertyFlags hostCoherent
            = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (std::uint32_t i = 0; i < sMaxFramesInFlight; ++i)
        {
            mFrameUbos[i] = createBuffer(sizeof(FrameUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, hostCoherent, true);

            mMaterialCapacities[i] = sInitialMaterialCap;
            mMaterialBuffers[i] = createBuffer(
                sizeof(GpuMaterial) * mMaterialCapacities[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostCoherent, true);

            mInstanceCapacities[i] = sInitialInstanceCap;
            mInstanceBuffers[i] = createBuffer(
                sizeof(GpuInstance) * mInstanceCapacities[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostCoherent, true);

            mLightCapacities[i] = sInitialLightCap;
            mLightBuffers[i] = createBuffer(
                sizeof(GpuPointLight) * mLightCapacities[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostCoherent, true);

            mBoneBuffers[i]
                = createBuffer(sizeof(float) * 16 * sMaxBones, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostCoherent, true);

            const VkDeviceSize indBytes = sizeof(VkDrawIndexedIndirectCommand) * mInstanceCapacities[i];
            mOpaqueIndirectBuffers[i]
                = createBuffer(indBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            mOpaqueCountBuffers[i] = createBuffer(sizeof(std::uint32_t),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
                    | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

            mTransparentIndirectBuffers[i]
                = createBuffer(indBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            mTransparentCountBuffers[i] = createBuffer(sizeof(std::uint32_t),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
                    | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        }

        recreateTileBuffers();
        for (std::uint32_t i = 0; i < sMaxFramesInFlight; ++i)
            updateFrameDescriptors(i);
        updatePostDescriptors();
    }

    void Renderer::recreateTileBuffers()
    {
        mTileCountX = std::max(1u, (mSwapchainExtent.width + 15u) / 16u);
        mTileCountY = std::max(1u, (mSwapchainExtent.height + 15u) / 16u);
        const VkDeviceSize tileBytes = sizeof(GpuTileLightList) * mTileCountX * mTileCountY;

        for (std::uint32_t i = 0; i < sMaxFramesInFlight; ++i)
        {
            destroyBuffer(mTileLightBuffers[i]);
            mTileLightBuffers[i]
                = createBuffer(tileBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        }
    }

    void Renderer::updateFrameDescriptors(std::uint32_t i)
    {
        VkDescriptorBufferInfo uboInfo{ mFrameUbos[i].mBuffer, 0, sizeof(FrameUniforms) };
        VkDescriptorBufferInfo matInfo{ mMaterialBuffers[i].mBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo instInfo{ mInstanceBuffers[i].mBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo lightInfo{ mLightBuffers[i].mBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo tileInfo{ mTileLightBuffers[i].mBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo boneInfo{ mBoneBuffers[i].mBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorImageInfo samplerInfo{ mLinearSampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
        VkDescriptorImageInfo shadowInfo{ mShadowSampler, mShadowArrayView,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo sceneDepthInfo{ mClampSampler, mResolvedDepthImage.mView,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

        VkDescriptorBufferInfo opIndInfo{ mOpaqueIndirectBuffers[i].mBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo opCntInfo{ mOpaqueCountBuffers[i].mBuffer, 0, sizeof(std::uint32_t) };
        VkDescriptorBufferInfo trIndInfo{ mTransparentIndirectBuffers[i].mBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo trCntInfo{ mTransparentCountBuffers[i].mBuffer, 0, sizeof(std::uint32_t) };

        std::array<VkWriteDescriptorSet, 14> writes{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mGlobalSets[i], 0, 0, 1,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uboInfo, nullptr };
        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mGlobalSets[i], 1, 0, 1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &matInfo, nullptr };
        writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mGlobalSets[i], 2, 0, 1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &instInfo, nullptr };
        writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mGlobalSets[i], 3, 0, 1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightInfo, nullptr };
        writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mGlobalSets[i], 4, 0, 1,
            VK_DESCRIPTOR_TYPE_SAMPLER, &samplerInfo, nullptr, nullptr };
        writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mGlobalSets[i], 6, 0, 1,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &shadowInfo, nullptr, nullptr };
        writes[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mGlobalSets[i], 7, 0, 1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &tileInfo, nullptr };
        writes[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mGlobalSets[i], 8, 0, 1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &boneInfo, nullptr };
        writes[8] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mGlobalSets[i], 9, 0, 1,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &sceneDepthInfo, nullptr, nullptr };

        writes[9] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mCullSets[i], 0, 0, 1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &opIndInfo, nullptr };
        writes[10] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mCullSets[i], 1, 0, 1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &opCntInfo, nullptr };
        writes[11] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mCullSets[i], 2, 0, 1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &trIndInfo, nullptr };
        writes[12] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mCullSets[i], 3, 0, 1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &trCntInfo, nullptr };
        writes[13] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mCullSets[i], 4, 0, 1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &tileInfo, nullptr };

        vkUpdateDescriptorSets(mDevice, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    void Renderer::updatePostDescriptors()
    {
        VkDescriptorImageInfo msaaDepthInfo{ mClampSampler, mMsaaDepthImage.mView,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo resDepthStorage{ VK_NULL_HANDLE, mResolvedDepthImage.mView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo resDepthSampled{ mClampSampler, mResolvedDepthImage.mView,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo aoStorage{ VK_NULL_HANDLE, mAoImage.mView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo aoSampled{ mClampSampler, mAoImage.mView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo sceneColorSampled{ mClampSampler, mResolvedColorImage.mView,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo bloomAStorage{ VK_NULL_HANDLE, mBloomA.mView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo bloomASampled{ mClampSampler, mBloomA.mView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo bloomBStorage{ VK_NULL_HANDLE, mBloomB.mView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo bloomBSampled{ mClampSampler, mBloomB.mView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

        std::array<VkWriteDescriptorSet, 11> writes{};
        // Depth resolve
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mDepthResolveSet, 0, 0, 1,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &msaaDepthInfo, nullptr, nullptr };
        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mDepthResolveSet, 1, 0, 1,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &resDepthStorage, nullptr, nullptr };
        // GTAO
        writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mGtaoSet, 0, 0, 1,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &resDepthSampled, nullptr, nullptr };
        writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mGtaoSet, 1, 0, 1,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &aoStorage, nullptr, nullptr };
        // Bloom downsample
        writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mBloomDownSet, 0, 0, 1,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &sceneColorSampled, nullptr, nullptr };
        writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mBloomDownSet, 1, 0, 1,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &bloomAStorage, nullptr, nullptr };
        // Bloom blur
        writes[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mBloomBlurSet, 0, 0, 1,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &bloomASampled, nullptr, nullptr };
        writes[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mBloomBlurSet, 1, 0, 1,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &bloomBStorage, nullptr, nullptr };
        // Final Post-Process
        writes[8] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mPostSet, 0, 0, 1,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &sceneColorSampled, nullptr, nullptr };
        writes[9] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mPostSet, 1, 0, 1,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &bloomBSampled, nullptr, nullptr };
        writes[10] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mPostSet, 2, 0, 1,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &aoSampled, nullptr, nullptr };

        vkUpdateDescriptorSets(mDevice, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    VkShaderModule Renderer::createShaderModule(const std::filesystem::path& spvPath) const
    {
        const std::vector<char> code = readBinaryFile(spvPath);
        VkShaderModuleCreateInfo createInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, code.size(),
            reinterpret_cast<const std::uint32_t*>(code.data()) };
        VkShaderModule mod = VK_NULL_HANDLE;
        vkCheck(vkCreateShaderModule(mDevice, &createInfo, nullptr, &mod), "vkCreateShaderModule");
        return mod;
    }

    void Renderer::createPipelines()
    {
        // Graphics + shadow layout (set = 0 + 4-byte cascade push constant for shadow)
        VkPushConstantRange shadowPush{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShadowPushConstants) };
        VkPipelineLayoutCreateInfo gfxLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1,
            &mGlobalSetLayout, 1, &shadowPush };
        vkCheck(vkCreatePipelineLayout(mDevice, &gfxLayoutInfo, nullptr, &mGraphicsPipelineLayout),
            "vkCreatePipelineLayout(gfx)");

        VkVertexInputBindingDescription binding{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX };
        std::array<VkVertexInputAttributeDescription, 6> attrs{};
        attrs[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<std::uint32_t>(offsetof(Vertex, mPos)) };
        attrs[1] = { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<std::uint32_t>(offsetof(Vertex, mNormal)) };
        attrs[2] = { 2, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<std::uint32_t>(offsetof(Vertex, mUV)) };
        attrs[3] = { 3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, static_cast<std::uint32_t>(offsetof(Vertex, mColor)) };
        attrs[4] = { 4, 0, VK_FORMAT_R32G32B32A32_UINT, static_cast<std::uint32_t>(offsetof(Vertex, mBoneIndices)) };
        attrs[5] = { 5, 0, VK_FORMAT_R32G32B32A32_SFLOAT, static_cast<std::uint32_t>(offsetof(Vertex, mBoneWeights)) };

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE
        };
        VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr,
            0, 1, nullptr, 1, nullptr };
        const std::array<VkDynamicState, 2> dynStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0,
            static_cast<std::uint32_t>(dynStates.size()), dynStates.data() };

        VkPipelineMultisampleStateCreateInfo msaaState{};
        msaaState.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        msaaState.rasterizationSamples = mMsaaSamples;

        VkPipelineMultisampleStateCreateInfo noMsaaState{};
        noMsaaState.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        noMsaaState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // Opaque + Transparent pipelines (MSAA)
        {
            VkShaderModule vertMod = createShaderModule(mShaderDir / "mesh.vert.spv");
            VkShaderModule fragMod = createShaderModule(mShaderDir / "mesh.frag.spv");
            std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
            stages[0] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT,
                vertMod, "main", nullptr };
            stages[1] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT,
                fragMod, "main", nullptr };

            VkPipelineVertexInputStateCreateInfo vertexInput{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                nullptr, 0, 1, &binding, static_cast<std::uint32_t>(attrs.size()), attrs.data() };
            VkPipelineRasterizationStateCreateInfo rasterizer{};
            rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
            rasterizer.lineWidth = 1.0f;
            rasterizer.cullMode = VK_CULL_MODE_NONE;
            rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

            VkPipelineDepthStencilStateCreateInfo depthStencil{};
            depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            depthStencil.depthTestEnable = VK_TRUE;
            depthStencil.depthWriteEnable = VK_TRUE;
            depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

            VkPipelineColorBlendAttachmentState blendAtt{};
            blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT
                | VK_COLOR_COMPONENT_A_BIT;
            VkPipelineColorBlendStateCreateInfo colorBlending{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 1, &blendAtt, { 0, 0, 0, 0 } };

            VkGraphicsPipelineCreateInfo pi{};
            pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pi.stageCount = static_cast<std::uint32_t>(stages.size());
            pi.pStages = stages.data();
            pi.pVertexInputState = &vertexInput;
            pi.pInputAssemblyState = &inputAssembly;
            pi.pViewportState = &viewportState;
            pi.pRasterizationState = &rasterizer;
            pi.pMultisampleState = &msaaState;
            pi.pDepthStencilState = &depthStencil;
            pi.pColorBlendState = &colorBlending;
            pi.pDynamicState = &dynamicState;
            pi.layout = mGraphicsPipelineLayout;
            pi.renderPass = mOpaqueRenderPass;
            pi.subpass = 0;

            vkCheck(vkCreateGraphicsPipelines(mDevice, VK_NULL_HANDLE, 1, &pi, nullptr, &mOpaquePipeline),
                "vkCreateGraphicsPipelines(opaque)");

            // Transparent / Water pipeline (on mTransparentRenderPass, MSAA)
            depthStencil.depthWriteEnable = VK_FALSE;
            blendAtt.blendEnable = VK_TRUE;
            blendAtt.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            blendAtt.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            blendAtt.colorBlendOp = VK_BLEND_OP_ADD;
            blendAtt.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            blendAtt.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            blendAtt.alphaBlendOp = VK_BLEND_OP_ADD;
            pi.renderPass = mTransparentRenderPass;

            vkCheck(vkCreateGraphicsPipelines(mDevice, VK_NULL_HANDLE, 1, &pi, nullptr, &mTransparentPipeline),
                "vkCreateGraphicsPipelines(transparent)");

            vkDestroyShaderModule(mDevice, fragMod, nullptr);
            vkDestroyShaderModule(mDevice, vertMod, nullptr);
        }

        // Procedural Sky pipeline (on mOpaqueRenderPass, MSAA)
        {
            VkShaderModule skyVert = createShaderModule(mShaderDir / "sky.vert.spv");
            VkShaderModule skyFrag = createShaderModule(mShaderDir / "sky.frag.spv");
            std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
            stages[0] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT,
                skyVert, "main", nullptr };
            stages[1] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT,
                skyFrag, "main", nullptr };

            VkPipelineVertexInputStateCreateInfo emptyVI{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                nullptr, 0, 0, nullptr, 0, nullptr };
            VkPipelineRasterizationStateCreateInfo rasterizer{};
            rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
            rasterizer.lineWidth = 1.0f;
            rasterizer.cullMode = VK_CULL_MODE_NONE;

            VkPipelineDepthStencilStateCreateInfo depthStencil{};
            depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            depthStencil.depthTestEnable = VK_TRUE;
            depthStencil.depthWriteEnable = VK_FALSE;
            depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

            VkPipelineColorBlendAttachmentState blendAtt{};
            blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT
                | VK_COLOR_COMPONENT_A_BIT;
            VkPipelineColorBlendStateCreateInfo colorBlending{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 1, &blendAtt, { 0, 0, 0, 0 } };

            VkGraphicsPipelineCreateInfo pi{};
            pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pi.stageCount = static_cast<std::uint32_t>(stages.size());
            pi.pStages = stages.data();
            pi.pVertexInputState = &emptyVI;
            pi.pInputAssemblyState = &inputAssembly;
            pi.pViewportState = &viewportState;
            pi.pRasterizationState = &rasterizer;
            pi.pMultisampleState = &msaaState;
            pi.pDepthStencilState = &depthStencil;
            pi.pColorBlendState = &colorBlending;
            pi.pDynamicState = &dynamicState;
            pi.layout = mGraphicsPipelineLayout;
            pi.renderPass = mOpaqueRenderPass;
            pi.subpass = 0;

            vkCheck(vkCreateGraphicsPipelines(mDevice, VK_NULL_HANDLE, 1, &pi, nullptr, &mSkyPipeline),
                "vkCreateGraphicsPipelines(sky)");

            vkDestroyShaderModule(mDevice, skyFrag, nullptr);
            vkDestroyShaderModule(mDevice, skyVert, nullptr);
        }

        // Shadow depth-only pipeline (1x)
        {
            VkShaderModule shadowVert = createShaderModule(mShaderDir / "shadow.vert.spv");
            VkPipelineShaderStageCreateInfo vertStage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                VK_SHADER_STAGE_VERTEX_BIT, shadowVert, "main", nullptr };
            const std::array<VkVertexInputAttributeDescription, 3> shadowAttrs = { attrs[0], attrs[4], attrs[5] };
            VkPipelineVertexInputStateCreateInfo vertexInput{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                nullptr, 0, 1, &binding, static_cast<std::uint32_t>(shadowAttrs.size()), shadowAttrs.data() };

            VkPipelineRasterizationStateCreateInfo rasterizer{};
            rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
            rasterizer.lineWidth = 1.0f;
            rasterizer.cullMode = VK_CULL_MODE_NONE;
            rasterizer.depthBiasEnable = VK_TRUE;
            rasterizer.depthBiasConstantFactor = 1.25f;
            rasterizer.depthBiasSlopeFactor = 1.75f;

            VkPipelineDepthStencilStateCreateInfo depthStencil{};
            depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            depthStencil.depthTestEnable = VK_TRUE;
            depthStencil.depthWriteEnable = VK_TRUE;
            depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

            VkGraphicsPipelineCreateInfo pi{};
            pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pi.stageCount = 1;
            pi.pStages = &vertStage;
            pi.pVertexInputState = &vertexInput;
            pi.pInputAssemblyState = &inputAssembly;
            pi.pViewportState = &viewportState;
            pi.pRasterizationState = &rasterizer;
            pi.pMultisampleState = &noMsaaState;
            pi.pDepthStencilState = &depthStencil;
            pi.pDynamicState = &dynamicState;
            pi.layout = mGraphicsPipelineLayout;
            pi.renderPass = mShadowRenderPass;
            pi.subpass = 0;

            vkCheck(vkCreateGraphicsPipelines(mDevice, VK_NULL_HANDLE, 1, &pi, nullptr, &mShadowPipeline),
                "vkCreateGraphicsPipelines(shadow)");
            vkDestroyShaderModule(mDevice, shadowVert, nullptr);
        }

        // Post-process pipeline (sets 0 + postSet, 1x SDR swapchain)
        {
            const std::array<VkDescriptorSetLayout, 2> postLayouts = { mGlobalSetLayout, mPostSetLayout };
            VkPipelineLayoutCreateInfo li{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0,
                static_cast<std::uint32_t>(postLayouts.size()), postLayouts.data(), 0, nullptr };
            vkCheck(vkCreatePipelineLayout(mDevice, &li, nullptr, &mPostLayout), "vkCreatePipelineLayout(post)");

            VkShaderModule pv = createShaderModule(mShaderDir / "post.vert.spv");
            VkShaderModule pf = createShaderModule(mShaderDir / "post.frag.spv");
            std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
            stages[0] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT,
                pv, "main", nullptr };
            stages[1] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT,
                pf, "main", nullptr };

            VkPipelineVertexInputStateCreateInfo emptyVI{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                nullptr, 0, 0, nullptr, 0, nullptr };
            VkPipelineRasterizationStateCreateInfo rasterizer{};
            rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
            rasterizer.lineWidth = 1.0f;
            rasterizer.cullMode = VK_CULL_MODE_NONE;

            VkPipelineColorBlendAttachmentState blendAtt{};
            blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT
                | VK_COLOR_COMPONENT_A_BIT;
            VkPipelineColorBlendStateCreateInfo colorBlending{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 1, &blendAtt, { 0, 0, 0, 0 } };

            VkGraphicsPipelineCreateInfo pi{};
            pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pi.stageCount = static_cast<std::uint32_t>(stages.size());
            pi.pStages = stages.data();
            pi.pVertexInputState = &emptyVI;
            pi.pInputAssemblyState = &inputAssembly;
            pi.pViewportState = &viewportState;
            pi.pRasterizationState = &rasterizer;
            pi.pMultisampleState = &noMsaaState;
            pi.pColorBlendState = &colorBlending;
            pi.pDynamicState = &dynamicState;
            pi.layout = mPostLayout;
            pi.renderPass = mPostRenderPass;
            pi.subpass = 0;

            vkCheck(vkCreateGraphicsPipelines(mDevice, VK_NULL_HANDLE, 1, &pi, nullptr, &mPostPipeline),
                "vkCreateGraphicsPipelines(post)");
            vkDestroyShaderModule(mDevice, pf, nullptr);
            vkDestroyShaderModule(mDevice, pv, nullptr);
        }

        // Compute pipelines: frustum cull, light cull, depth resolve, GTAO, bloom
        {
            const std::array<VkDescriptorSetLayout, 2> cullSetLayouts = { mGlobalSetLayout, mCullSetLayout };
            VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CullPushConstants) };
            VkPipelineLayoutCreateInfo li{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0,
                static_cast<std::uint32_t>(cullSetLayouts.size()), cullSetLayouts.data(), 1, &pushRange };
            vkCheck(
                vkCreatePipelineLayout(mDevice, &li, nullptr, &mCullPipelineLayout), "vkCreatePipelineLayout(cull)");

            auto makeCompute = [&](const char* spv, VkPipelineLayout layout, VkPipeline& out) {
                VkShaderModule mod = createShaderModule(mShaderDir / spv);
                VkComputePipelineCreateInfo ci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0,
                    { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, mod,
                        "main", nullptr },
                    layout, VK_NULL_HANDLE, 0 };
                vkCheck(vkCreateComputePipelines(mDevice, VK_NULL_HANDLE, 1, &ci, nullptr, &out), spv);
                vkDestroyShaderModule(mDevice, mod, nullptr);
            };

            makeCompute("cull.comp.spv", mCullPipelineLayout, mCullPipeline);
            makeCompute("light_cull.comp.spv", mCullPipelineLayout, mLightCullPipeline);

            // Depth resolve layout
            VkPushConstantRange drPush{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DepthResolvePushConstants) };
            VkPipelineLayoutCreateInfo drLi{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1,
                &mCompute2ImgSetLayout, 1, &drPush };
            vkCheck(vkCreatePipelineLayout(mDevice, &drLi, nullptr, &mDepthResolveLayout),
                "vkCreatePipelineLayout(depthResolve)");
            makeCompute("depth_resolve.comp.spv", mDepthResolveLayout, mDepthResolvePipeline);

            // GTAO layout (globalSet + compute2ImgSet)
            const std::array<VkDescriptorSetLayout, 2> gtaoLayouts = { mGlobalSetLayout, mCompute2ImgSetLayout };
            VkPipelineLayoutCreateInfo gtaoLi{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0,
                static_cast<std::uint32_t>(gtaoLayouts.size()), gtaoLayouts.data(), 0, nullptr };
            vkCheck(vkCreatePipelineLayout(mDevice, &gtaoLi, nullptr, &mGtaoLayout), "vkCreatePipelineLayout(gtao)");
            makeCompute("gtao.comp.spv", mGtaoLayout, mGtaoPipeline);

            // Bloom layout
            VkPushConstantRange bloomPush{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BloomPushConstants) };
            VkPipelineLayoutCreateInfo bloomLi{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1,
                &mCompute2ImgSetLayout, 1, &bloomPush };
            vkCheck(vkCreatePipelineLayout(mDevice, &bloomLi, nullptr, &mBloomLayout), "vkCreatePipelineLayout(bloom)");
            makeCompute("bloom.comp.spv", mBloomLayout, mBloomPipeline);
        }
    }

    void Renderer::createCommandBuffers()
    {
        VkCommandBufferAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, mCommandPool,
            VK_COMMAND_BUFFER_LEVEL_PRIMARY, sMaxFramesInFlight };
        vkCheck(vkAllocateCommandBuffers(mDevice, &allocInfo, mCommandBuffers.data()), "vkAllocateCommandBuffers");
    }

    void Renderer::createSyncObjects()
    {
        VkSemaphoreCreateInfo semInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, nullptr, 0 };
        VkFenceCreateInfo fenceInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, VK_FENCE_CREATE_SIGNALED_BIT };

        for (std::uint32_t i = 0; i < sMaxFramesInFlight; ++i)
        {
            vkCheck(vkCreateSemaphore(mDevice, &semInfo, nullptr, &mImageAvailableSemaphores[i]), "vkCreateSemaphore");
            vkCheck(vkCreateFence(mDevice, &fenceInfo, nullptr, &mInFlightFences[i]), "vkCreateFence");
        }

        mRenderFinishedSemaphores.resize(mSwapchainImages.size(), VK_NULL_HANDLE);
        for (VkSemaphore& sem : mRenderFinishedSemaphores)
            vkCheck(vkCreateSemaphore(mDevice, &semInfo, nullptr, &sem), "vkCreateSemaphore(present)");
    }

    void Renderer::createDefaultResources()
    {
        uploadTexture(makeSolidColorImage(255, 255, 255, 255));
        addMaterial(GpuMaterial{});
        mStagedBones.push_back(osg::Matrixf::identity());
    }

    std::uint32_t Renderer::findMemoryType(std::uint32_t typeFilter, VkMemoryPropertyFlags properties) const
    {
        VkPhysicalDeviceMemoryProperties memProps{};
        vkGetPhysicalDeviceMemoryProperties(mPhysicalDevice, &memProps);
        for (std::uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
        {
            if ((typeFilter & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties)
                return i;
        }
        throw std::runtime_error("Failed to find suitable Vulkan memory type");
    }

    Renderer::AllocatedBuffer Renderer::createBuffer(
        VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, bool persistentMap)
    {
        AllocatedBuffer buf;
        buf.mSize = std::max<VkDeviceSize>(size, 16);
        VkBufferCreateInfo bufInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, buf.mSize, usage,
            VK_SHARING_MODE_EXCLUSIVE, 0, nullptr };
        vkCheck(vkCreateBuffer(mDevice, &bufInfo, nullptr, &buf.mBuffer), "vkCreateBuffer");

        VkMemoryRequirements memReq{};
        vkGetBufferMemoryRequirements(mDevice, buf.mBuffer, &memReq);

        VkMemoryAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, memReq.size,
            findMemoryType(memReq.memoryTypeBits, properties) };
        vkCheck(vkAllocateMemory(mDevice, &allocInfo, nullptr, &buf.mMemory), "vkAllocateMemory");
        vkCheck(vkBindBufferMemory(mDevice, buf.mBuffer, buf.mMemory, 0), "vkBindBufferMemory");

        if (persistentMap)
            vkCheck(vkMapMemory(mDevice, buf.mMemory, 0, buf.mSize, 0, &buf.mMapped), "vkMapMemory(persistent)");
        return buf;
    }

    void Renderer::destroyBuffer(AllocatedBuffer& buf)
    {
        if (buf.mMapped != nullptr && buf.mMemory != VK_NULL_HANDLE)
        {
            vkUnmapMemory(mDevice, buf.mMemory);
            buf.mMapped = nullptr;
        }
        if (buf.mBuffer != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(mDevice, buf.mBuffer, nullptr);
            buf.mBuffer = VK_NULL_HANDLE;
        }
        if (buf.mMemory != VK_NULL_HANDLE)
        {
            vkFreeMemory(mDevice, buf.mMemory, nullptr);
            buf.mMemory = VK_NULL_HANDLE;
        }
        buf.mSize = 0;
    }

    VkCommandBuffer Renderer::beginSingleTimeCommands()
    {
        VkCommandBufferAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, mCommandPool,
            VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        vkCheck(vkAllocateCommandBuffers(mDevice, &allocInfo, &cmd), "vkAllocateCommandBuffers(oneshot)");
        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr };
        vkCheck(vkBeginCommandBuffer(cmd, &beginInfo), "vkBeginCommandBuffer(oneshot)");
        return cmd;
    }

    void Renderer::endSingleTimeCommands(VkCommandBuffer cmd)
    {
        vkCheck(vkEndCommandBuffer(cmd), "vkEndCommandBuffer(oneshot)");
        VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cmd, 0, nullptr };
        vkCheck(vkQueueSubmit(mGraphicsQueue, 1, &submitInfo, VK_NULL_HANDLE), "vkQueueSubmit(oneshot)");
        vkCheck(vkQueueWaitIdle(mGraphicsQueue), "vkQueueWaitIdle(oneshot)");
        vkFreeCommandBuffers(mDevice, mCommandPool, 1, &cmd);
    }

    void Renderer::uploadToDeviceBuffer(
        const void* data, VkDeviceSize size, VkBufferUsageFlags usage, AllocatedBuffer& out)
    {
        destroyBuffer(out);
        const VkDeviceSize allocSize = std::max<VkDeviceSize>(size, 16);
        AllocatedBuffer staging = createBuffer(allocSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);
        if (data != nullptr && size > 0)
            std::memcpy(staging.mMapped, data, static_cast<std::size_t>(size));
        else
            std::memset(staging.mMapped, 0, static_cast<std::size_t>(allocSize));

        out = createBuffer(allocSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        VkCommandBuffer cmd = beginSingleTimeCommands();
        VkBufferCopy region{ 0, 0, allocSize };
        vkCmdCopyBuffer(cmd, staging.mBuffer, out.mBuffer, 1, &region);
        endSingleTimeCommands(cmd);

        destroyBuffer(staging);
    }

    void Renderer::transitionImageLayout(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout,
        VkImageLayout newLayout, std::uint32_t mipLevels, VkImageAspectFlags aspect, std::uint32_t layerCount)
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = { aspect, 0, mipLevels, 0, layerCount };

        VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        {
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
            && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_GENERAL)
        {
            barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_GENERAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
            && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.dstAccessMask
                = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
            && (newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                || newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL))
        {
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        }

        vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    TextureId Renderer::uploadTexture(const DecodedImage& image)
    {
        if (!image.valid() || mTextures.size() >= sMaxBindlessTextures)
            return 0;

        const std::uint32_t baseW = image.mMips[0].mWidth;
        const std::uint32_t baseH = image.mMips[0].mHeight;
        const bool generateMips = !image.mCompressed && image.mMips.size() == 1 && (baseW > 1 || baseH > 1);
        const std::uint32_t mipLevels = generateMips
            ? (static_cast<std::uint32_t>(std::floor(std::log2(std::max(baseW, baseH)))) + 1u)
            : static_cast<std::uint32_t>(image.mMips.size());

        AllocatedImage tex = createImage2D(baseW, baseH, image.mFormat,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                | (generateMips ? static_cast<VkImageUsageFlags>(VK_IMAGE_USAGE_TRANSFER_SRC_BIT) : 0u),
            VK_IMAGE_ASPECT_COLOR_BIT, VK_SAMPLE_COUNT_1_BIT, mipLevels);

        VkDeviceSize totalSize = 0;
        for (const auto& mip : image.mMips)
            totalSize += mip.mData.size();

        AllocatedBuffer staging = createBuffer(totalSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);

        std::vector<VkBufferImageCopy> copies;
        VkDeviceSize offset = 0;
        for (std::uint32_t m = 0; m < image.mMips.size(); ++m)
        {
            const auto& mip = image.mMips[m];
            std::memcpy(static_cast<std::byte*>(staging.mMapped) + offset, mip.mData.data(), mip.mData.size());
            VkBufferImageCopy region{};
            region.bufferOffset = offset;
            region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, m, 0, 1 };
            region.imageExtent = { mip.mWidth, mip.mHeight, 1 };
            copies.push_back(region);
            offset += mip.mData.size();
        }

        VkCommandBuffer cmd = beginSingleTimeCommands();
        transitionImageLayout(
            cmd, tex.mImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
        vkCmdCopyBufferToImage(cmd, staging.mBuffer, tex.mImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            static_cast<std::uint32_t>(copies.size()), copies.data());

        if (generateMips)
        {
            std::int32_t mipW = static_cast<std::int32_t>(baseW);
            std::int32_t mipH = static_cast<std::int32_t>(baseH);
            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.image = tex.mImage;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            for (std::uint32_t i = 1; i < mipLevels; ++i)
            {
                barrier.subresourceRange.baseMipLevel = i - 1;
                barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                    0, nullptr, 1, &barrier);

                VkImageBlit blit{};
                blit.srcOffsets[1] = { mipW, mipH, 1 };
                blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1 };
                blit.dstOffsets[1] = { std::max(1, mipW / 2), std::max(1, mipH / 2), 1 };
                blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1 };
                vkCmdBlitImage(cmd, tex.mImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, tex.mImage,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

                barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                    nullptr, 0, nullptr, 1, &barrier);

                mipW = std::max(1, mipW / 2);
                mipH = std::max(1, mipH / 2);
            }

            barrier.subresourceRange.baseMipLevel = mipLevels - 1;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                nullptr, 0, nullptr, 1, &barrier);
        }
        else
        {
            transitionImageLayout(cmd, tex.mImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, mipLevels);
        }
        endSingleTimeCommands(cmd);
        destroyBuffer(staging);

        const TextureId id = static_cast<TextureId>(mTextures.size());
        mTextures.push_back(tex);

        VkDescriptorImageInfo imgDesc{ VK_NULL_HANDLE, tex.mView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        for (std::uint32_t i = 0; i < sMaxFramesInFlight; ++i)
        {
            VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mGlobalSets[i], 5, id, 1,
                VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &imgDesc, nullptr, nullptr };
            vkUpdateDescriptorSets(mDevice, 1, &write, 0, nullptr);
        }
        return id;
    }

    MaterialId Renderer::addMaterial(const GpuMaterial& material)
    {
        const MaterialId id = static_cast<MaterialId>(mStagedMaterials.size());
        mStagedMaterials.push_back(material);
        mMaterialsDirty.fill(true);
        return id;
    }

    MeshId Renderer::addMesh(const MeshData& mesh, const std::vector<MaterialId>& resolvedMaterials)
    {
        if (mesh.mVertices.empty() || mesh.mIndices.empty())
            return sInvalidHandle;

        const std::int32_t baseVertex = static_cast<std::int32_t>(mStagedVertices.size());
        const std::uint32_t baseIndex = static_cast<std::uint32_t>(mStagedIndices.size());

        mStagedVertices.insert(mStagedVertices.end(), mesh.mVertices.begin(), mesh.mVertices.end());
        mStagedIndices.insert(mStagedIndices.end(), mesh.mIndices.begin(), mesh.mIndices.end());
        mGeometryDirty = true;

        StagedMesh staged;
        if (!mesh.mSubmeshes.empty())
        {
            for (const auto& sub : mesh.mSubmeshes)
            {
                StagedSubmesh s;
                s.mFirstIndex = baseIndex + sub.mFirstIndex;
                s.mIndexCount = sub.mIndexCount;
                s.mVertexOffset = baseVertex;
                s.mMaterialId
                    = (sub.mMaterialIndex < resolvedMaterials.size()) ? resolvedMaterials[sub.mMaterialIndex] : 0;
                if (s.mMaterialId < mStagedMaterials.size())
                    s.mTransparent
                        = (mStagedMaterials[s.mMaterialId].mFlags & (MaterialFlag_Water | MaterialFlag_Transparent))
                        != 0;
                s.mBoundsCenter = sub.mBoundsCenter;
                s.mBoundsRadius = sub.mBoundsRadius;
                staged.mSubmeshes.push_back(s);
            }
        }
        else
        {
            StagedSubmesh s;
            s.mFirstIndex = baseIndex;
            s.mIndexCount = static_cast<std::uint32_t>(mesh.mIndices.size());
            s.mVertexOffset = baseVertex;
            s.mMaterialId = resolvedMaterials.empty() ? 0 : resolvedMaterials[0];
            s.mBoundsCenter = (mesh.mBoundsMin + mesh.mBoundsMax) * 0.5f;
            s.mBoundsRadius = std::max(((mesh.mBoundsMax - mesh.mBoundsMin) * 0.5f).length(), 0.1f);
            staged.mSubmeshes.push_back(s);
        }

        const MeshId id = static_cast<MeshId>(mStagedMeshes.size());
        mStagedMeshes.push_back(std::move(staged));
        return id;
    }

    std::uint32_t Renderer::getMeshSubmeshCount(MeshId meshId) const
    {
        if (meshId >= mStagedMeshes.size())
            return 0;
        return static_cast<std::uint32_t>(mStagedMeshes[meshId].mSubmeshes.size());
    }

    InstanceId Renderer::addInstance(MeshId meshId, const osg::Matrixf& worldMatrix, std::uint32_t boneOffset)
    {
        if (meshId >= mStagedMeshes.size())
            return sInvalidHandle;

        const auto& submeshes = mStagedMeshes[meshId].mSubmeshes;
        const float maxScale = computeMaxScale(worldMatrix);
        InstanceId firstId = sInvalidHandle;

        if (submeshes.size() == 1 && !mFreeInstanceSlots.empty())
        {
            firstId = mFreeInstanceSlots.back();
            mFreeInstanceSlots.pop_back();
        }
        else
        {
            firstId = static_cast<InstanceId>(mStagedInstances.size());
            mStagedInstances.resize(mStagedInstances.size() + submeshes.size());
        }

        for (std::size_t i = 0; i < submeshes.size(); ++i)
        {
            const auto& sub = submeshes[i];
            GpuInstance& inst = mStagedInstances[firstId + i];
            std::memcpy(inst.mModel, worldMatrix.ptr(), sizeof(inst.mModel));
            const osg::Vec3f worldCenter = sub.mBoundsCenter * worldMatrix;
            inst.mBoundsSphere[0] = worldCenter.x();
            inst.mBoundsSphere[1] = worldCenter.y();
            inst.mBoundsSphere[2] = worldCenter.z();
            inst.mBoundsSphere[3] = sub.mBoundsRadius * maxScale;
            inst.mFirstIndex = sub.mFirstIndex;
            inst.mIndexCount = sub.mIndexCount;
            inst.mVertexOffset = sub.mVertexOffset;
            inst.mMaterialId = sub.mMaterialId;
            inst.mBoneOffset = boneOffset;
            inst.mFlags = sub.mTransparent ? InstanceFlag_Transparent : InstanceFlag_None;
        }
        mInstancesDirty.fill(true);
        return firstId;
    }

    void Renderer::updateInstanceTransform(InstanceId firstId, MeshId meshId, const osg::Matrixf& worldMatrix)
    {
        if (meshId >= mStagedMeshes.size())
            return;
        const auto& submeshes = mStagedMeshes[meshId].mSubmeshes;
        if (firstId + submeshes.size() > mStagedInstances.size())
            return;

        const float maxScale = computeMaxScale(worldMatrix);
        for (std::size_t i = 0; i < submeshes.size(); ++i)
        {
            GpuInstance& inst = mStagedInstances[firstId + i];
            std::memcpy(inst.mModel, worldMatrix.ptr(), sizeof(inst.mModel));
            const osg::Vec3f worldCenter = submeshes[i].mBoundsCenter * worldMatrix;
            inst.mBoundsSphere[0] = worldCenter.x();
            inst.mBoundsSphere[1] = worldCenter.y();
            inst.mBoundsSphere[2] = worldCenter.z();
            inst.mBoundsSphere[3] = submeshes[i].mBoundsRadius * maxScale;
        }
        mInstancesDirty.fill(true);
    }

    void Renderer::removeInstance(InstanceId firstId, std::uint32_t submeshCount)
    {
        for (std::uint32_t i = 0; i < submeshCount; ++i)
        {
            const std::uint32_t slot = firstId + i;
            if (slot < mStagedInstances.size())
            {
                mStagedInstances[slot].mIndexCount = 0;
                mFreeInstanceSlots.push_back(slot);
            }
        }
        mInstancesDirty.fill(true);
    }

    LightId Renderer::addPointLight(const GpuPointLight& light)
    {
        const LightId id = static_cast<LightId>(mStagedLights.size());
        mStagedLights.push_back(light);
        mLightsDirty.fill(true);
        return id;
    }

    void Renderer::updatePointLight(LightId id, const GpuPointLight& light)
    {
        if (id >= mStagedLights.size())
            return;
        mStagedLights[id] = light;
        mLightsDirty.fill(true);
    }

    void Renderer::removePointLight(LightId id)
    {
        if (id >= mStagedLights.size())
            return;
        mStagedLights[id].mPosRadius[3] = 0.0f;
        mStagedLights[id].mColorIntensity[3] = 0.0f;
        mLightsDirty.fill(true);
    }

    std::uint32_t Renderer::allocateBonePalette(std::uint32_t boneCount)
    {
        if (mStagedBones.size() + boneCount > sMaxBones)
            return sInvalidHandle;
        const std::uint32_t offset = static_cast<std::uint32_t>(mStagedBones.size());
        mStagedBones.resize(mStagedBones.size() + boneCount, osg::Matrixf::identity());
        mBonesDirty.fill(true);
        return offset;
    }

    void Renderer::updateBonePalette(std::uint32_t boneOffset, std::span<const osg::Matrixf> bones)
    {
        if (boneOffset == sInvalidHandle || boneOffset + bones.size() > mStagedBones.size())
            return;
        std::copy(bones.begin(), bones.end(), mStagedBones.begin() + boneOffset);
        mBonesDirty.fill(true);
    }

    void Renderer::clearScene()
    {
        vkDeviceWaitIdle(mDevice);
        mStagedVertices.clear();
        mStagedIndices.clear();
        mStagedMeshes.clear();
        mStagedInstances.clear();
        mFreeInstanceSlots.clear();
        mStagedLights.clear();
        mStagedBones.assign(1, osg::Matrixf::identity());
        if (mStagedMaterials.size() > 1)
            mStagedMaterials.resize(1);
        mGeometryDirty = false;
        mMaterialsDirty.fill(true);
        mInstancesDirty.fill(true);
        mLightsDirty.fill(true);
        mBonesDirty.fill(true);
    }

    SceneStats Renderer::getSceneStats() const
    {
        SceneStats s;
        s.mMeshCount = mStagedMeshes.size();
        s.mTextureCount = mTextures.size();
        s.mMaterialCount = mStagedMaterials.size();
        s.mInstanceCount = mStagedInstances.size();
        s.mVertexCount = mStagedVertices.size();
        s.mTriangleCount = mStagedIndices.size() / 3;
        s.mPointLightCount = mStagedLights.size();
        return s;
    }

    void Renderer::commitScene()
    {
        if (!mGeometryDirty)
            return;
        vkDeviceWaitIdle(mDevice);
        if (!mStagedVertices.empty())
            uploadToDeviceBuffer(mStagedVertices.data(), sizeof(Vertex) * mStagedVertices.size(),
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mVertexBuffer);
        if (!mStagedIndices.empty())
            uploadToDeviceBuffer(mStagedIndices.data(), sizeof(std::uint32_t) * mStagedIndices.size(),
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT, mIndexBuffer);
        mGeometryDirty = false;
    }

    void Renderer::ensureDynamicBufferCapacities(std::uint32_t i)
    {
        const VkMemoryPropertyFlags hostCoherent
            = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        bool descriptorDirty = false;

        if (mStagedMaterials.size() > mMaterialCapacities[i])
        {
            while (mMaterialCapacities[i] < mStagedMaterials.size())
                mMaterialCapacities[i] *= 2;
            destroyBuffer(mMaterialBuffers[i]);
            mMaterialBuffers[i] = createBuffer(
                sizeof(GpuMaterial) * mMaterialCapacities[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostCoherent, true);
            mMaterialsDirty[i] = true;
            descriptorDirty = true;
        }

        if (mStagedInstances.size() > mInstanceCapacities[i])
        {
            while (mInstanceCapacities[i] < mStagedInstances.size())
                mInstanceCapacities[i] *= 2;
            destroyBuffer(mInstanceBuffers[i]);
            mInstanceBuffers[i] = createBuffer(
                sizeof(GpuInstance) * mInstanceCapacities[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostCoherent, true);

            const VkDeviceSize indBytes = sizeof(VkDrawIndexedIndirectCommand) * mInstanceCapacities[i];
            destroyBuffer(mOpaqueIndirectBuffers[i]);
            mOpaqueIndirectBuffers[i]
                = createBuffer(indBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            destroyBuffer(mTransparentIndirectBuffers[i]);
            mTransparentIndirectBuffers[i]
                = createBuffer(indBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

            mInstancesDirty[i] = true;
            descriptorDirty = true;
        }

        if (mStagedLights.size() > mLightCapacities[i])
        {
            while (mLightCapacities[i] < mStagedLights.size())
                mLightCapacities[i] *= 2;
            destroyBuffer(mLightBuffers[i]);
            mLightBuffers[i] = createBuffer(
                sizeof(GpuPointLight) * mLightCapacities[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostCoherent, true);
            mLightsDirty[i] = true;
            descriptorDirty = true;
        }

        if (descriptorDirty)
            updateFrameDescriptors(i);
    }

    void Renderer::syncDynamicBuffers(std::uint32_t i)
    {
        ensureDynamicBufferCapacities(i);

        if (mMaterialsDirty[i] && !mStagedMaterials.empty())
        {
            std::memcpy(
                mMaterialBuffers[i].mMapped, mStagedMaterials.data(), sizeof(GpuMaterial) * mStagedMaterials.size());
            mMaterialsDirty[i] = false;
        }
        if (mInstancesDirty[i] && !mStagedInstances.empty())
        {
            std::memcpy(
                mInstanceBuffers[i].mMapped, mStagedInstances.data(), sizeof(GpuInstance) * mStagedInstances.size());
            mInstancesDirty[i] = false;
        }
        if (mLightsDirty[i] && !mStagedLights.empty())
        {
            std::memcpy(mLightBuffers[i].mMapped, mStagedLights.data(), sizeof(GpuPointLight) * mStagedLights.size());
            mLightsDirty[i] = false;
        }
        if (mBonesDirty[i] && !mStagedBones.empty())
        {
            std::memcpy(mBoneBuffers[i].mMapped, mStagedBones.data(), sizeof(float) * 16 * mStagedBones.size());
            mBonesDirty[i] = false;
        }
    }

    void Renderer::cleanupSwapchain()
    {
        for (VkFramebuffer fb : mSwapchainFramebuffers)
            vkDestroyFramebuffer(mDevice, fb, nullptr);
        mSwapchainFramebuffers.clear();

        if (mTransparentFramebuffer != VK_NULL_HANDLE)
        {
            vkDestroyFramebuffer(mDevice, mTransparentFramebuffer, nullptr);
            mTransparentFramebuffer = VK_NULL_HANDLE;
        }
        if (mOpaqueFramebuffer != VK_NULL_HANDLE)
        {
            vkDestroyFramebuffer(mDevice, mOpaqueFramebuffer, nullptr);
            mOpaqueFramebuffer = VK_NULL_HANDLE;
        }

        destroyImage(mBloomB);
        destroyImage(mBloomA);
        destroyImage(mAoImage);
        destroyImage(mResolvedDepthImage);
        destroyImage(mResolvedColorImage);
        destroyImage(mMsaaDepthImage);
        destroyImage(mMsaaColorImage);

        for (VkImageView view : mSwapchainImageViews)
            vkDestroyImageView(mDevice, view, nullptr);
        mSwapchainImageViews.clear();

        if (mSwapchain != VK_NULL_HANDLE)
        {
            vkDestroySwapchainKHR(mDevice, mSwapchain, nullptr);
            mSwapchain = VK_NULL_HANDLE;
        }
    }

    void Renderer::recreateSwapchain()
    {
        int w = 0;
        int h = 0;
        SDL_Vulkan_GetDrawableSize(mWindow, &w, &h);
        if (w == 0 || h == 0)
            return;

        vkDeviceWaitIdle(mDevice);
        cleanupSwapchain();
        createSwapchain();
        createOffscreenTargets();
        createFramebuffers();
        recreateTileBuffers();
        for (std::uint32_t i = 0; i < sMaxFramesInFlight; ++i)
            updateFrameDescriptors(i);
        updatePostDescriptors();

        if (mRenderFinishedSemaphores.size() != mSwapchainImages.size())
        {
            for (VkSemaphore sem : mRenderFinishedSemaphores)
                vkDestroySemaphore(mDevice, sem, nullptr);
            mRenderFinishedSemaphores.assign(mSwapchainImages.size(), VK_NULL_HANDLE);
            VkSemaphoreCreateInfo semInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, nullptr, 0 };
            for (VkSemaphore& sem : mRenderFinishedSemaphores)
                vkCheck(vkCreateSemaphore(mDevice, &semInfo, nullptr, &sem), "vkCreateSemaphore(recreate)");
        }
    }

    void Renderer::drawFrame(const FrameView& view)
    {
        if (mSwapchainExtent.width == 0 || mSwapchainExtent.height == 0)
        {
            recreateSwapchain();
            return;
        }

        if (mGeometryDirty)
            commitScene();

        vkCheck(vkWaitForFences(mDevice, 1, &mInFlightFences[mCurrentFrame], VK_TRUE, UINT64_MAX), "vkWaitForFences");

        std::uint32_t imageIndex = 0;
        VkResult acq = vkAcquireNextImageKHR(
            mDevice, mSwapchain, UINT64_MAX, mImageAvailableSemaphores[mCurrentFrame], VK_NULL_HANDLE, &imageIndex);
        if (acq == VK_ERROR_OUT_OF_DATE_KHR)
        {
            recreateSwapchain();
            return;
        }
        if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR)
            vkCheck(acq, "vkAcquireNextImageKHR");

        vkCheck(vkResetFences(mDevice, 1, &mInFlightFences[mCurrentFrame]), "vkResetFences");

        syncDynamicBuffers(mCurrentFrame);

        FrameUniforms ubo{};
        std::memcpy(ubo.mView, view.mView.ptr(), sizeof(ubo.mView));
        std::memcpy(ubo.mProj, view.mProj.ptr(), sizeof(ubo.mProj));
        std::memcpy(ubo.mViewProj, view.mViewProj.ptr(), sizeof(ubo.mViewProj));
        const osg::Matrixf invProj = osg::Matrixf::inverse(view.mProj);
        std::memcpy(ubo.mInvProj, invProj.ptr(), sizeof(ubo.mInvProj));
        for (std::uint32_t c = 0; c < sShadowCascadeCount; ++c)
            std::memcpy(ubo.mCascadeViewProj[c], view.mCascadeViewProj[c].ptr(), sizeof(ubo.mCascadeViewProj[c]));
        ubo.mCascadeSplits[0] = view.mCascadeSplits.x();
        ubo.mCascadeSplits[1] = view.mCascadeSplits.y();
        ubo.mCascadeSplits[2] = view.mCascadeSplits.z();
        ubo.mCameraPos[0] = view.mCameraPos.x();
        ubo.mCameraPos[1] = view.mCameraPos.y();
        ubo.mCameraPos[2] = view.mCameraPos.z();
        ubo.mCameraPos[3] = view.mTime;
        ubo.mSunDir[0] = view.mSunDir.x();
        ubo.mSunDir[1] = view.mSunDir.y();
        ubo.mSunDir[2] = view.mSunDir.z();
        ubo.mSunDir[3] = 0.0018f;
        ubo.mSunColor[0] = view.mSunColor.x();
        ubo.mSunColor[1] = view.mSunColor.y();
        ubo.mSunColor[2] = view.mSunColor.z();
        ubo.mSunColor[3] = view.mSunColor.w();
        ubo.mAmbientColor[0] = view.mAmbientColor.x();
        ubo.mAmbientColor[1] = view.mAmbientColor.y();
        ubo.mAmbientColor[2] = view.mAmbientColor.z();
        ubo.mAmbientColor[3] = view.mAmbientColor.w();
        ubo.mFogColor[0] = view.mFogColor.x();
        ubo.mFogColor[1] = view.mFogColor.y();
        ubo.mFogColor[2] = view.mFogColor.z();
        ubo.mFogColor[3] = view.mFogStart;
        ubo.mFogParams[0] = view.mFogEnd;
        ubo.mFogParams[1] = static_cast<float>(mStagedLights.size());
        ubo.mFogParams[2] = view.mMaxDrawDistance;
        ubo.mFogParams[3] = view.mEnableShadows ? 1.0f : 0.0f;
        ubo.mScreenParams[0] = static_cast<float>(mSwapchainExtent.width);
        ubo.mScreenParams[1] = static_cast<float>(mSwapchainExtent.height);
        ubo.mScreenParams[2] = static_cast<float>(mTileCountX);
        ubo.mScreenParams[3] = static_cast<float>(mTileCountY);
        ubo.mVisualParams[0] = view.mEnableSky ? 1.0f : 0.0f;
        ubo.mVisualParams[1] = view.mEnableGTAO ? 1.0f : 0.0f;
        ubo.mVisualParams[2] = view.mEnableBloom ? 1.0f : 0.0f;
        ubo.mVisualParams[3] = view.mEnableToneMap ? 1.0f : 0.0f;
        extractFrustumPlanes(view.mViewProj, ubo.mFrustumPlanes);
        std::memcpy(mFrameUbos[mCurrentFrame].mMapped, &ubo, sizeof(ubo));

        VkCommandBuffer cmd = mCommandBuffers[mCurrentFrame];
        vkCheck(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, 0, nullptr };
        vkCheck(vkBeginCommandBuffer(cmd, &beginInfo), "vkBeginCommandBuffer");

        const std::uint32_t instanceCount = static_cast<std::uint32_t>(mStagedInstances.size());
        const bool hasGeometry
            = instanceCount > 0 && mVertexBuffer.mBuffer != VK_NULL_HANDLE && mIndexBuffer.mBuffer != VK_NULL_HANDLE;

        auto dispatchCull = [&](bool enableFrustumCull, bool opaqueOnly) {
            vkCmdFillBuffer(cmd, mOpaqueCountBuffers[mCurrentFrame].mBuffer, 0, sizeof(std::uint32_t), 0);
            vkCmdFillBuffer(cmd, mTransparentCountBuffers[mCurrentFrame].mBuffer, 0, sizeof(std::uint32_t), 0);

            std::array<VkBufferMemoryBarrier, 2> fillBarriers{};
            fillBarriers[0] = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED, mOpaqueCountBuffers[mCurrentFrame].mBuffer, 0, sizeof(std::uint32_t) };
            fillBarriers[1] = fillBarriers[0];
            fillBarriers[1].buffer = mTransparentCountBuffers[mCurrentFrame].mBuffer;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                nullptr, static_cast<std::uint32_t>(fillBarriers.size()), fillBarriers.data(), 0, nullptr);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mCullPipeline);
            const std::array<VkDescriptorSet, 2> sets = { mGlobalSets[mCurrentFrame], mCullSets[mCurrentFrame] };
            vkCmdBindDescriptorSets(
                cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mCullPipelineLayout, 0, 2, sets.data(), 0, nullptr);

            CullPushConstants pc{ instanceCount, enableFrustumCull ? 1u : 0u, opaqueOnly ? 1u : 0u };
            vkCmdPushConstants(cmd, mCullPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, (instanceCount + 63u) / 64u, 1, 1);

            std::array<VkBufferMemoryBarrier, 4> postCull{};
            postCull[0] = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_INDIRECT_COMMAND_READ_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                mOpaqueIndirectBuffers[mCurrentFrame].mBuffer, 0, VK_WHOLE_SIZE };
            postCull[1] = postCull[0];
            postCull[1].buffer = mOpaqueCountBuffers[mCurrentFrame].mBuffer;
            postCull[1].size = sizeof(std::uint32_t);
            postCull[2] = postCull[0];
            postCull[2].buffer = mTransparentIndirectBuffers[mCurrentFrame].mBuffer;
            postCull[3] = postCull[0];
            postCull[3].buffer = mTransparentCountBuffers[mCurrentFrame].mBuffer;
            postCull[3].size = sizeof(std::uint32_t);

            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 0,
                nullptr, static_cast<std::uint32_t>(postCull.size()), postCull.data(), 0, nullptr);
        };

        // Pass 1: 3-Cascade Directional Shadow Maps (MDI)
        if (hasGeometry && view.mEnableShadows)
        {
            dispatchCull(false, true);

            VkViewport vp{ 0.f, 0.f, static_cast<float>(mShadowMapSize), static_cast<float>(mShadowMapSize), 0.f, 1.f };
            VkRect2D sc{ { 0, 0 }, { mShadowMapSize, mShadowMapSize } };
            VkClearValue depthClear{};
            depthClear.depthStencil = { 1.0f, 0 };
            const VkBuffer vbs[] = { mVertexBuffer.mBuffer };
            const VkDeviceSize offsets[] = { 0 };

            for (std::uint32_t c = 0; c < sShadowCascadeCount; ++c)
            {
                VkRenderPassBeginInfo shadowBegin{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, nullptr, mShadowRenderPass,
                    mShadowFramebuffers[c], { { 0, 0 }, { mShadowMapSize, mShadowMapSize } }, 1, &depthClear };
                vkCmdBeginRenderPass(cmd, &shadowBegin, VK_SUBPASS_CONTENTS_INLINE);
                vkCmdSetViewport(cmd, 0, 1, &vp);
                vkCmdSetScissor(cmd, 0, 1, &sc);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mShadowPipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mGraphicsPipelineLayout, 0, 1,
                    &mGlobalSets[mCurrentFrame], 0, nullptr);
                ShadowPushConstants spc{ c };
                vkCmdPushConstants(cmd, mGraphicsPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(spc), &spc);
                vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);
                vkCmdBindIndexBuffer(cmd, mIndexBuffer.mBuffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexedIndirectCount(cmd, mOpaqueIndirectBuffers[mCurrentFrame].mBuffer, 0,
                    mOpaqueCountBuffers[mCurrentFrame].mBuffer, 0, instanceCount, sizeof(VkDrawIndexedIndirectCommand));
                vkCmdEndRenderPass(cmd);
            }

            std::array<VkBufferMemoryBarrier, 2> reuseBarrier{};
            reuseBarrier[0] = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr, VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
                VK_ACCESS_SHADER_WRITE_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                mOpaqueIndirectBuffers[mCurrentFrame].mBuffer, 0, VK_WHOLE_SIZE };
            reuseBarrier[1] = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr, VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                mOpaqueCountBuffers[mCurrentFrame].mBuffer, 0, sizeof(std::uint32_t) };
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                static_cast<std::uint32_t>(reuseBarrier.size()), reuseBarrier.data(), 0, nullptr);
        }

        // Pass 2a: Main Camera 2-Bucket Frustum Cull
        if (hasGeometry)
            dispatchCull(view.mEnableFrustumCull, false);

        // Pass 2b: Forward+ Tiled Light Culling
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mLightCullPipeline);
            const std::array<VkDescriptorSet, 2> sets = { mGlobalSets[mCurrentFrame], mCullSets[mCurrentFrame] };
            vkCmdBindDescriptorSets(
                cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mCullPipelineLayout, 0, 2, sets.data(), 0, nullptr);
            vkCmdDispatch(cmd, (mTileCountX + 7u) / 8u, (mTileCountY + 7u) / 8u, 1);

            VkBufferMemoryBarrier tileBarrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                mTileLightBuffers[mCurrentFrame].mBuffer, 0, VK_WHOLE_SIZE };
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                nullptr, 1, &tileBarrier, 0, nullptr);
        }

        const VkViewport viewport{ 0.f, 0.f, static_cast<float>(mSwapchainExtent.width),
            static_cast<float>(mSwapchainExtent.height), 0.f, 1.f };
        const VkRect2D scissor{ { 0, 0 }, mSwapchainExtent };
        const VkBuffer vbs[] = { mVertexBuffer.mBuffer };
        const VkDeviceSize offsets[] = { 0 };

        // Pass 3: Opaque MDI + Procedural Sky (MSAA Float Target)
        {
            std::array<VkClearValue, 2> clears{};
            clears[0].color = { { view.mFogColor.x(), view.mFogColor.y(), view.mFogColor.z(), 1.0f } };
            clears[1].depthStencil = { 1.0f, 0 };
            VkRenderPassBeginInfo rpBegin{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, nullptr, mOpaqueRenderPass,
                mOpaqueFramebuffer, scissor, static_cast<std::uint32_t>(clears.size()), clears.data() };
            vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            vkCmdSetScissor(cmd, 0, 1, &scissor);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mGraphicsPipelineLayout, 0, 1,
                &mGlobalSets[mCurrentFrame], 0, nullptr);

            if (hasGeometry)
            {
                vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);
                vkCmdBindIndexBuffer(cmd, mIndexBuffer.mBuffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mOpaquePipeline);
                vkCmdDrawIndexedIndirectCount(cmd, mOpaqueIndirectBuffers[mCurrentFrame].mBuffer, 0,
                    mOpaqueCountBuffers[mCurrentFrame].mBuffer, 0, instanceCount, sizeof(VkDrawIndexedIndirectCommand));
            }
            if (view.mEnableSky)
            {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mSkyPipeline);
                vkCmdDraw(cmd, 3, 1, 0, 0);
            }
            vkCmdEndRenderPass(cmd);
        }

        // Pass 4a: Resolve MSAA Depth -> 1x R32_SFLOAT for GTAO + Depth-Aware Water
        {
            transitionImageLayout(cmd, mResolvedDepthImage.mImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mDepthResolvePipeline);
            vkCmdBindDescriptorSets(
                cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mDepthResolveLayout, 0, 1, &mDepthResolveSet, 0, nullptr);
            DepthResolvePushConstants dpc{ mSwapchainExtent.width, mSwapchainExtent.height,
                static_cast<std::uint32_t>(mMsaaSamples) };
            vkCmdPushConstants(cmd, mDepthResolveLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(dpc), &dpc);
            vkCmdDispatch(cmd, (mSwapchainExtent.width + 7u) / 8u, (mSwapchainExtent.height + 7u) / 8u, 1);
            transitionImageLayout(
                cmd, mResolvedDepthImage.mImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            transitionImageLayout(cmd, mMsaaDepthImage.mImage, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1, VK_IMAGE_ASPECT_DEPTH_BIT);
        }

        // Pass 4b: Compute GTAO (Half-Res Ambient Occlusion)
        const std::uint32_t halfW = std::max(1u, mSwapchainExtent.width / 2u);
        const std::uint32_t halfH = std::max(1u, mSwapchainExtent.height / 2u);
        {
            transitionImageLayout(cmd, mAoImage.mImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mGtaoPipeline);
            const std::array<VkDescriptorSet, 2> gtaoSets = { mGlobalSets[mCurrentFrame], mGtaoSet };
            vkCmdBindDescriptorSets(
                cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mGtaoLayout, 0, 2, gtaoSets.data(), 0, nullptr);
            vkCmdDispatch(cmd, (halfW + 7u) / 8u, (halfH + 7u) / 8u, 1);
            transitionImageLayout(
                cmd, mAoImage.mImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }

        // Pass 5: Depth-Aware Transparent / Water MDI + MSAA Resolve -> mResolvedColorImage
        {
            VkRenderPassBeginInfo rpBegin{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, nullptr, mTransparentRenderPass,
                mTransparentFramebuffer, scissor, 0, nullptr };
            vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            vkCmdSetScissor(cmd, 0, 1, &scissor);

            if (hasGeometry)
            {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mGraphicsPipelineLayout, 0, 1,
                    &mGlobalSets[mCurrentFrame], 0, nullptr);
                vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);
                vkCmdBindIndexBuffer(cmd, mIndexBuffer.mBuffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mTransparentPipeline);
                vkCmdDrawIndexedIndirectCount(cmd, mTransparentIndirectBuffers[mCurrentFrame].mBuffer, 0,
                    mTransparentCountBuffers[mCurrentFrame].mBuffer, 0, instanceCount,
                    sizeof(VkDrawIndexedIndirectCommand));
            }
            vkCmdEndRenderPass(cmd);
        }

        // Pass 6: Half-Res Compute Bloom (Bright Threshold -> 9-Tap Tent Blur)
        if (view.mEnableBloom)
        {
            transitionImageLayout(cmd, mBloomA.mImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mBloomPipeline);
            vkCmdBindDescriptorSets(
                cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mBloomLayout, 0, 1, &mBloomDownSet, 0, nullptr);
            BloomPushConstants bp0{ 0u, 1.05f };
            vkCmdPushConstants(cmd, mBloomLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(bp0), &bp0);
            vkCmdDispatch(cmd, (halfW + 7u) / 8u, (halfH + 7u) / 8u, 1);
            transitionImageLayout(
                cmd, mBloomA.mImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

            transitionImageLayout(cmd, mBloomB.mImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
            vkCmdBindDescriptorSets(
                cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mBloomLayout, 0, 1, &mBloomBlurSet, 0, nullptr);
            BloomPushConstants bp1{ 1u, 1.05f };
            vkCmdPushConstants(cmd, mBloomLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(bp1), &bp1);
            vkCmdDispatch(cmd, (halfW + 7u) / 8u, (halfH + 7u) / 8u, 1);
            transitionImageLayout(
                cmd, mBloomB.mImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }

        // Pass 7: Final Post-Process Pass -> SDR Swapchain (GTAO + Bloom + CAS + ACES Tone Map)
        {
            VkRenderPassBeginInfo rpBegin{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, nullptr, mPostRenderPass,
                mSwapchainFramebuffers[imageIndex], scissor, 0, nullptr };
            vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            vkCmdSetScissor(cmd, 0, 1, &scissor);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mPostPipeline);
            const std::array<VkDescriptorSet, 2> postSets = { mGlobalSets[mCurrentFrame], mPostSet };
            vkCmdBindDescriptorSets(
                cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mPostLayout, 0, 2, postSets.data(), 0, nullptr);
            vkCmdDraw(cmd, 3, 1, 0, 0);
            vkCmdEndRenderPass(cmd);
        }

        vkCheck(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 1, &mImageAvailableSemaphores[mCurrentFrame],
            &waitStage, 1, &cmd, 1, &mRenderFinishedSemaphores[imageIndex] };
        vkCheck(vkQueueSubmit(mGraphicsQueue, 1, &submitInfo, mInFlightFences[mCurrentFrame]), "vkQueueSubmit");

        VkPresentInfoKHR presentInfo{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, nullptr, 1,
            &mRenderFinishedSemaphores[imageIndex], 1, &mSwapchain, &imageIndex, nullptr };
        VkResult presentRes = vkQueuePresentKHR(mPresentQueue, &presentInfo);
        if (presentRes == VK_ERROR_OUT_OF_DATE_KHR || presentRes == VK_SUBOPTIMAL_KHR || mFramebufferResized)
        {
            mFramebufferResized = false;
            recreateSwapchain();
        }
        else if (presentRes != VK_SUCCESS)
        {
            vkCheck(presentRes, "vkQueuePresentKHR");
        }

        mCurrentFrame = (mCurrentFrame + 1) % sMaxFramesInFlight;
    }
}
