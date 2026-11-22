#version 450

layout(location = 0) out vec2 outNDC;

void main()
{
    // Fullscreen triangle at far plane z = 1.0
    vec2 positions[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    vec2 pos = positions[gl_VertexIndex];
    outNDC = pos;
    gl_Position = vec4(pos, 1.0, 1.0);
}
