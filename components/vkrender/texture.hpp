#ifndef OPENMW_COMPONENTS_VKRENDER_TEXTURE_HPP
#define OPENMW_COMPONENTS_VKRENDER_TEXTURE_HPP

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace VkRender
{
    struct DecodedMip
    {
        std::uint32_t mWidth{ 0 };
        std::uint32_t mHeight{ 0 };
        std::vector<std::byte> mData;
    };

    struct DecodedImage
    {
        VkFormat mFormat{ VK_FORMAT_R8G8B8A8_SRGB };
        bool mCompressed{ false };
        std::vector<DecodedMip> mMips;

        bool valid() const { return !mMips.empty() && mMips[0].mWidth > 0 && mMips[0].mHeight > 0; }
    };

    /// Decode .dds (BC1/BC2/BC3/RGBA/BGRA), .tga, .png, or .jpg/.jpeg from memory into a Vulkan-ready image.
    DecodedImage decodeImage(std::span<const std::byte> bytes, std::string_view extensionHint);

    /// Create a 1x1 solid RGBA8 image (used as bindless slot 0 white fallback).
    DecodedImage makeSolidColorImage(std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a);
}

#endif
