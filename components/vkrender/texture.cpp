#include "texture.hpp"

#include <components/misc/strings/algorithm.hpp>

#include <jpeglib.h>
#include <png.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace VkRender
{
    namespace
    {
        std::uint32_t readU32(const std::byte* p)
        {
            std::uint32_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return v;
        }

        std::uint16_t readU16(const std::byte* p)
        {
            std::uint16_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return v;
        }

        DecodedImage decodeDDS(std::span<const std::byte> bytes)
        {
            if (bytes.size() < 128 || std::memcmp(bytes.data(), "DDS ", 4) != 0)
                return {};

            const std::byte* hdr = bytes.data() + 4;
            const std::uint32_t height = readU32(hdr + 8);
            const std::uint32_t width = readU32(hdr + 12);
            std::uint32_t mipCount = std::max(1u, readU32(hdr + 24));

            const std::byte* pf = hdr + 72;
            const std::uint32_t pfFlags = readU32(pf + 4);
            const std::uint32_t fourCC = readU32(pf + 8);
            const std::uint32_t rgbBits = readU32(pf + 12);
            const std::uint32_t rMask = readU32(pf + 16);

            DecodedImage img;
            std::size_t blockSize = 0;

            constexpr std::uint32_t DDPF_FOURCC = 0x4;
            constexpr std::uint32_t FOURCC_DXT1 = 0x31545844;
            constexpr std::uint32_t FOURCC_DXT3 = 0x33545844;
            constexpr std::uint32_t FOURCC_DXT5 = 0x35545844;

            std::size_t offset = 128;
            if ((pfFlags & DDPF_FOURCC) != 0)
            {
                img.mCompressed = true;
                if (fourCC == FOURCC_DXT1)
                {
                    img.mFormat = VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
                    blockSize = 8;
                }
                else if (fourCC == FOURCC_DXT3)
                {
                    img.mFormat = VK_FORMAT_BC2_SRGB_BLOCK;
                    blockSize = 16;
                }
                else if (fourCC == FOURCC_DXT5)
                {
                    img.mFormat = VK_FORMAT_BC3_SRGB_BLOCK;
                    blockSize = 16;
                }
                else
                {
                    return {};
                }
            }
            else if (rgbBits == 32)
            {
                img.mCompressed = false;
                img.mFormat = (rMask == 0x000000FFu) ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_B8G8R8A8_SRGB;
            }
            else
            {
                return {};
            }

            std::uint32_t w = width;
            std::uint32_t h = height;
            for (std::uint32_t m = 0; m < mipCount; ++m)
            {
                const std::size_t mipSize = img.mCompressed
                    ? (static_cast<std::size_t>(std::max(1u, (w + 3u) / 4u))
                          * static_cast<std::size_t>(std::max(1u, (h + 3u) / 4u)) * blockSize)
                    : (static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4u);

                if (offset + mipSize > bytes.size())
                    break;

                DecodedMip mip;
                mip.mWidth = w;
                mip.mHeight = h;
                mip.mData.resize(mipSize);
                std::memcpy(mip.mData.data(), bytes.data() + offset, mipSize);
                img.mMips.push_back(std::move(mip));

                offset += mipSize;
                w = std::max(1u, w >> 1u);
                h = std::max(1u, h >> 1u);
            }
            return img;
        }

        DecodedImage decodeTGA(std::span<const std::byte> bytes)
        {
            if (bytes.size() < 18)
                return {};

            const auto* d = reinterpret_cast<const std::uint8_t*>(bytes.data());
            const std::uint8_t idLen = d[0];
            const std::uint8_t colorMapType = d[1];
            const std::uint8_t imageType = d[2];
            const std::uint16_t width = readU16(bytes.data() + 12);
            const std::uint16_t height = readU16(bytes.data() + 14);
            const std::uint8_t bpp = d[16];
            const std::uint8_t desc = d[17];

            if (colorMapType != 0 || (imageType != 2 && imageType != 10) || (bpp != 24 && bpp != 32) || width == 0
                || height == 0)
                return {};

            const std::size_t pixelCount = static_cast<std::size_t>(width) * height;
            const std::size_t bytesPerPixel = bpp / 8;
            std::size_t src = 18 + idLen;
            if (src > bytes.size())
                return {};

            std::vector<std::uint8_t> rgba(pixelCount * 4u);
            auto writePixel = [&](std::size_t idx, const std::uint8_t* p) {
                rgba[idx * 4 + 0] = p[2];
                rgba[idx * 4 + 1] = p[1];
                rgba[idx * 4 + 2] = p[0];
                rgba[idx * 4 + 3] = (bytesPerPixel == 4) ? p[3] : 255;
            };

            if (imageType == 2)
            {
                if (src + pixelCount * bytesPerPixel > bytes.size())
                    return {};
                for (std::size_t i = 0; i < pixelCount; ++i)
                    writePixel(i, d + src + i * bytesPerPixel);
            }
            else
            {
                std::size_t outIdx = 0;
                while (outIdx < pixelCount && src < bytes.size())
                {
                    const std::uint8_t header = d[src++];
                    const std::size_t count = (header & 0x7Fu) + 1u;
                    if ((header & 0x80u) != 0)
                    {
                        if (src + bytesPerPixel > bytes.size())
                            return {};
                        const std::uint8_t* p = d + src;
                        src += bytesPerPixel;
                        for (std::size_t c = 0; c < count && outIdx < pixelCount; ++c)
                            writePixel(outIdx++, p);
                    }
                    else
                    {
                        if (src + count * bytesPerPixel > bytes.size())
                            return {};
                        for (std::size_t c = 0; c < count && outIdx < pixelCount; ++c)
                        {
                            writePixel(outIdx++, d + src);
                            src += bytesPerPixel;
                        }
                    }
                }
            }

            const bool topOrigin = (desc & 0x20u) != 0;
            if (!topOrigin)
            {
                const std::size_t rowBytes = static_cast<std::size_t>(width) * 4u;
                std::vector<std::uint8_t> tmp(rowBytes);
                for (std::uint16_t y = 0; y < height / 2; ++y)
                {
                    std::uint8_t* rowA = rgba.data() + static_cast<std::size_t>(y) * rowBytes;
                    std::uint8_t* rowB = rgba.data() + static_cast<std::size_t>(height - 1 - y) * rowBytes;
                    std::memcpy(tmp.data(), rowA, rowBytes);
                    std::memcpy(rowA, rowB, rowBytes);
                    std::memcpy(rowB, tmp.data(), rowBytes);
                }
            }

            DecodedImage img;
            img.mFormat = VK_FORMAT_R8G8B8A8_SRGB;
            DecodedMip mip;
            mip.mWidth = width;
            mip.mHeight = height;
            mip.mData.resize(rgba.size());
            std::memcpy(mip.mData.data(), rgba.data(), rgba.size());
            img.mMips.push_back(std::move(mip));
            return img;
        }

        struct PngMemReader
        {
            const std::byte* mData;
            std::size_t mSize;
            std::size_t mOffset;
        };

        void pngReadCallback(png_structp pngPtr, png_bytep outBytes, png_size_t byteCount)
        {
            auto* reader = static_cast<PngMemReader*>(png_get_io_ptr(pngPtr));
            if (reader->mOffset + byteCount > reader->mSize)
                png_error(pngPtr, "Unexpected EOF in PNG stream");
            std::memcpy(outBytes, reader->mData + reader->mOffset, byteCount);
            reader->mOffset += byteCount;
        }

        DecodedImage decodePNG(std::span<const std::byte> bytes)
        {
            if (bytes.size() < 8 || png_sig_cmp(reinterpret_cast<png_const_bytep>(bytes.data()), 0, 8) != 0)
                return {};

            png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
            if (!png)
                return {};
            png_infop info = png_create_info_struct(png);
            if (!info)
            {
                png_destroy_read_struct(&png, nullptr, nullptr);
                return {};
            }

            if (setjmp(png_jmpbuf(png)))
            {
                png_destroy_read_struct(&png, &info, nullptr);
                return {};
            }

            PngMemReader reader{ bytes.data(), bytes.size(), 0 };
            png_set_read_fn(png, &reader, pngReadCallback);
            png_read_info(png, info);

            const png_uint_32 width = png_get_image_width(png, info);
            const png_uint_32 height = png_get_image_height(png, info);
            const png_byte colorType = png_get_color_type(png, info);
            const png_byte bitDepth = png_get_bit_depth(png, info);

            if (bitDepth == 16)
                png_set_strip_16(png);
            if (colorType == PNG_COLOR_TYPE_PALETTE)
                png_set_palette_to_rgb(png);
            if (colorType == PNG_COLOR_TYPE_GRAY && bitDepth < 8)
                png_set_expand_gray_1_2_4_to_8(png);
            if (png_get_valid(png, info, PNG_INFO_tRNS))
                png_set_tRNS_to_alpha(png);
            if (colorType == PNG_COLOR_TYPE_RGB || colorType == PNG_COLOR_TYPE_GRAY
                || colorType == PNG_COLOR_TYPE_PALETTE)
                png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
            if (colorType == PNG_COLOR_TYPE_GRAY || colorType == PNG_COLOR_TYPE_GRAY_ALPHA)
                png_set_gray_to_rgb(png);

            png_read_update_info(png, info);

            DecodedMip mip;
            mip.mWidth = width;
            mip.mHeight = height;
            const std::size_t rowBytes = static_cast<std::size_t>(width) * 4u;
            mip.mData.resize(rowBytes * height);

            std::vector<png_bytep> rows(height);
            for (png_uint_32 y = 0; y < height; ++y)
                rows[y] = reinterpret_cast<png_bytep>(mip.mData.data() + static_cast<std::size_t>(y) * rowBytes);

            png_read_image(png, rows.data());
            png_destroy_read_struct(&png, &info, nullptr);

            DecodedImage img;
            img.mFormat = VK_FORMAT_R8G8B8A8_SRGB;
            img.mMips.push_back(std::move(mip));
            return img;
        }

        DecodedImage decodeJPEG(std::span<const std::byte> bytes)
        {
            if (bytes.size() < 4)
                return {};

            jpeg_decompress_struct cinfo{};
            jpeg_error_mgr jerr{};
            cinfo.err = jpeg_std_error(&jerr);
            jpeg_create_decompress(&cinfo);

            jpeg_mem_src(
                &cinfo, reinterpret_cast<const unsigned char*>(bytes.data()), static_cast<unsigned long>(bytes.size()));
            if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK)
            {
                jpeg_destroy_decompress(&cinfo);
                return {};
            }

            cinfo.out_color_space = JCS_RGB;
            jpeg_start_decompress(&cinfo);

            const std::uint32_t width = cinfo.output_width;
            const std::uint32_t height = cinfo.output_height;
            std::vector<unsigned char> rowBuf(static_cast<std::size_t>(width) * 3u);

            DecodedMip mip;
            mip.mWidth = width;
            mip.mHeight = height;
            mip.mData.resize(static_cast<std::size_t>(width) * height * 4u);

            while (cinfo.output_scanline < height)
            {
                const std::uint32_t y = cinfo.output_scanline;
                unsigned char* rowPtr = rowBuf.data();
                jpeg_read_scanlines(&cinfo, &rowPtr, 1);
                auto* dst
                    = reinterpret_cast<std::uint8_t*>(mip.mData.data() + static_cast<std::size_t>(y) * width * 4u);
                for (std::uint32_t x = 0; x < width; ++x)
                {
                    dst[x * 4 + 0] = rowBuf[x * 3 + 0];
                    dst[x * 4 + 1] = rowBuf[x * 3 + 1];
                    dst[x * 4 + 2] = rowBuf[x * 3 + 2];
                    dst[x * 4 + 3] = 255;
                }
            }

            jpeg_finish_decompress(&cinfo);
            jpeg_destroy_decompress(&cinfo);

            DecodedImage img;
            img.mFormat = VK_FORMAT_R8G8B8A8_SRGB;
            img.mMips.push_back(std::move(mip));
            return img;
        }
    }

    DecodedImage decodeImage(std::span<const std::byte> bytes, std::string_view extensionHint)
    {
        if (bytes.empty())
            return {};

        if (bytes.size() >= 4 && std::memcmp(bytes.data(), "DDS ", 4) == 0)
            return decodeDDS(bytes);
        if (bytes.size() >= 8 && png_sig_cmp(reinterpret_cast<png_const_bytep>(bytes.data()), 0, 8) == 0)
            return decodePNG(bytes);
        if (bytes.size() >= 2 && std::to_integer<std::uint8_t>(bytes[0]) == 0xFF
            && std::to_integer<std::uint8_t>(bytes[1]) == 0xD8)
            return decodeJPEG(bytes);

        if (Misc::StringUtils::ciEndsWith(extensionHint, ".tga"))
            return decodeTGA(bytes);
        if (Misc::StringUtils::ciEndsWith(extensionHint, ".dds"))
            return decodeDDS(bytes);
        if (Misc::StringUtils::ciEndsWith(extensionHint, ".png"))
            return decodePNG(bytes);
        if (Misc::StringUtils::ciEndsWith(extensionHint, ".jpg")
            || Misc::StringUtils::ciEndsWith(extensionHint, ".jpeg"))
            return decodeJPEG(bytes);

        return decodeTGA(bytes);
    }

    DecodedImage makeSolidColorImage(std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a)
    {
        DecodedImage img;
        img.mFormat = VK_FORMAT_R8G8B8A8_SRGB;
        DecodedMip mip;
        mip.mWidth = 1;
        mip.mHeight = 1;
        mip.mData = { static_cast<std::byte>(r), static_cast<std::byte>(g), static_cast<std::byte>(b),
            static_cast<std::byte>(a) };
        img.mMips.push_back(std::move(mip));
        return img;
    }
}
