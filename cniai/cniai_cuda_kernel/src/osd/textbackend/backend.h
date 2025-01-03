#pragma once

#include <memory>
#include <tuple>
#include <vector>

// #define ENABLE_TEXT_BACKEND_PANGO
// #define ENABLE_TEXT_BACKEND_STB
#define MAX_FONT_SIZE 200

namespace cniai_cuda_kernel::cuosd {

enum class TextBackendType : int { None = 0, PangoCairo = 1, StbTrueType = 2 };

class WordMeta {
public:
    virtual int width() const = 0;
    virtual int height() const = 0;
    virtual int x_offset_on_bitmap() const = 0;
    virtual int xadvance(int font_size, bool empty = false) const = 0;
};

class WordMetaMapper {
public:
    virtual WordMeta *query(unsigned long int word) = 0;
};

class TextBackend {
public:
    virtual std::vector<unsigned long int> split_utf8(const char *utf8_text) = 0;
    virtual std::tuple<int, int, int>
    measure_text(const std::vector<unsigned long int> &words, unsigned int font_size,
                 const char *font) = 0;
    virtual void add_build_text(const std::vector<unsigned long int> &words,
                                unsigned int font_size, const char *font) = 0;
    virtual void build_bitmap(void *stream = nullptr) = 0;
    virtual WordMetaMapper *query(const char *font, int font_size) = 0;
    virtual unsigned char *bitmap_device_pointer() const = 0;
    virtual int bitmap_width() const = 0;
    virtual int compute_y_offset(int max_glyph_height, int h, WordMeta *word,
                                 int font_size) const = 0;
    virtual int uniform_font_size(int size) const = 0;
};

const char *text_backend_type_name(TextBackendType backend);
std::shared_ptr<TextBackend> create_text_backend(TextBackendType backend);
std::string concat_font_name_size(const char *name, int size);

} // namespace cniai_cuda_kernel::cuosd