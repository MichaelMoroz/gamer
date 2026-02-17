static const float PI = 3.14159265;

struct RenderUniforms
{
    static const uint MAX_COMPONENTS = 64;

    uint size;
    uint component_count;
    float exposure;
    float gamma;
    float saturation;
    float ray_step;

    float3 camera;

    float3 axis;
    float winding_b;
    float winding_n;
    float no_arms;
    float arm1;
    float arm2;
    float arm3;
    float arm4;

    float4x4 inv_vp;

    int class_id[MAX_COMPONENTS];
    float4 comp_p0[MAX_COMPONENTS];
    float4 comp_p1[MAX_COMPONENTS];
    float4 comp_p2[MAX_COMPONENTS];
    float4 comp_spec[MAX_COMPONENTS];
};

cbuffer Uniforms
{
    RenderUniforms U;
}

RWStructuredBuffer<uint> out_pixels;
RWTexture2D<float4> out_image;

static const int grad3[12][3] = {
    {1,1,0}, {-1,1,0}, {1,-1,0}, {-1,-1,0},
    {1,0,1}, {-1,0,1}, {1,0,-1}, {-1,0,-1},
    {0,1,1}, {0,-1,1}, {0,1,-1}, {0,-1,-1}
};

static const int perm[512] = {
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
    35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
    18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
    250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
    189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
    172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
    228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
    107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,

    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
    35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
    18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
    250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
    189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
    172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
    228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
    107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
};

int fastfloor(float x)
{
    return x > 0.0 ? int(x) : int(x) - 1;
}

float dot3i(int3 g, float x, float y, float z)
{
    return float(g.x) * x + float(g.y) * y + float(g.z) * z;
}

float raw_noise_3d(float x, float y, float z)
{
    float n0, n1, n2, n3;
    float F3 = 1.0 / 3.0;
    float s = (x + y + z) * F3;
    int i = fastfloor(x + s);
    int j = fastfloor(y + s);
    int k = fastfloor(z + s);

    float G3 = 1.0 / 6.0;
    float t = (i + j + k) * G3;
    float X0 = i - t;
    float Y0 = j - t;
    float Z0 = k - t;
    float x0 = x - X0;
    float y0 = y - Y0;
    float z0 = z - Z0;

    int i1, j1, k1;
    int i2, j2, k2;

    if (x0 >= y0)
    {
        if (y0 >= z0) { i1=1; j1=0; k1=0; i2=1; j2=1; k2=0; }
        else if (x0 >= z0) { i1=1; j1=0; k1=0; i2=1; j2=0; k2=1; }
        else { i1=0; j1=0; k1=1; i2=1; j2=0; k2=1; }
    }
    else
    {
        if (y0 < z0) { i1=0; j1=0; k1=1; i2=0; j2=1; k2=1; }
        else if (x0 < z0) { i1=0; j1=1; k1=0; i2=0; j2=1; k2=1; }
        else { i1=0; j1=1; k1=0; i2=1; j2=1; k2=0; }
    }

    float x1 = x0 - i1 + G3;
    float y1 = y0 - j1 + G3;
    float z1 = z0 - k1 + G3;
    float x2 = x0 - i2 + 2.0 * G3;
    float y2 = y0 - j2 + 2.0 * G3;
    float z2 = z0 - k2 + 2.0 * G3;
    float x3 = x0 - 1.0 + 3.0 * G3;
    float y3 = y0 - 1.0 + 3.0 * G3;
    float z3 = z0 - 1.0 + 3.0 * G3;

    int ii = i & 255;
    int jj = j & 255;
    int kk = k & 255;
    int gi0 = perm[ii + perm[jj + perm[kk]]] % 12;
    int gi1 = perm[ii + i1 + perm[jj + j1 + perm[kk + k1]]] % 12;
    int gi2 = perm[ii + i2 + perm[jj + j2 + perm[kk + k2]]] % 12;
    int gi3 = perm[ii + 1 + perm[jj + 1 + perm[kk + 1]]] % 12;

    float t0 = 0.6 - x0*x0 - y0*y0 - z0*z0;
    if (t0 < 0.0) n0 = 0.0;
    else { t0 *= t0; n0 = t0*t0*dot3i(int3(grad3[gi0][0], grad3[gi0][1], grad3[gi0][2]), x0, y0, z0); }

    float t1 = 0.6 - x1*x1 - y1*y1 - z1*z1;
    if (t1 < 0.0) n1 = 0.0;
    else { t1 *= t1; n1 = t1*t1*dot3i(int3(grad3[gi1][0], grad3[gi1][1], grad3[gi1][2]), x1, y1, z1); }

    float t2 = 0.6 - x2*x2 - y2*y2 - z2*z2;
    if (t2 < 0.0) n2 = 0.0;
    else { t2 *= t2; n2 = t2*t2*dot3i(int3(grad3[gi2][0], grad3[gi2][1], grad3[gi2][2]), x2, y2, z2); }

    float t3 = 0.6 - x3*x3 - y3*y3 - z3*z3;
    if (t3 < 0.0) n3 = 0.0;
    else { t3 *= t3; n3 = t3*t3*dot3i(int3(grad3[gi3][0], grad3[gi3][1], grad3[gi3][2]), x3, y3, z3); }

    return 32.0 * (n0 + n1 + n2 + n3);
}

float octave_noise_3d(float octaves, float persistence, float scale, float x, float y, float z)
{
    float total = 0.0;
    float frequency = scale;
    float amplitude = 1.0;
    float maxAmplitude = 0.0;
    for (int i = 0; i < int(octaves); i++)
    {
        total += raw_noise_3d(x * frequency, y * frequency, z * frequency) * amplitude;
        frequency *= 2.0;
        maxAmplitude += amplitude;
        amplitude *= persistence;
    }
    return total / maxAmplitude;
}

float get_ridged_mf(float3 p, float frequency, int octaves, float lacunarity, float offset, float gain)
{
    float value = 0.0;
    float weight = 1.0;
    float w = -0.05;
    float3 vt = p;
    float freq = frequency;
    for (int octave = 0; octave < octaves; octave++)
    {
        float signal = raw_noise_3d(vt.x, vt.y, vt.z);
        signal = abs(signal);
        signal = offset - signal;
        signal *= signal;
        signal *= weight;
        weight = signal * gain;
        if (weight > 1.0) weight = 1.0;
        if (weight < 0.0) weight = 0.0;
        value += signal * pow(freq, w);
        vt *= lacunarity;
        freq *= lacunarity;
    }
    return (value * 1.25) - 1.0;
}

bool intersect_sphere(float3 o, float3 d, float3 r, out float3 isp1, out float3 isp2, out float t0, out float t1)
{
    float3 inv = float3(1.0 / (r.x * r.x), 1.0 / (r.y * r.y), 1.0 / (r.z * r.z));
    float3 rD = d * inv;
    float3 rO = o * inv;

    float A = dot(d, rD);
    float B = 2.0 * dot(d, rO);
    float C = dot(o, rO) - 1.0;
    float S = B*B - 4.0*A*C;
    if (S <= 0.0)
    {
        isp1 = 0;
        isp2 = 0;
        t0 = 0;
        t1 = 0;
        return false;
    }

    t0 = (-B - sqrt(S)) / (2.0 * A);
    t1 = (-B + sqrt(S)) / (2.0 * A);
    isp1 = o + d * t0;
    isp2 = o + d * t1;
    return true;
}

float3 floor_vec(float3 v)
{
    return float3(max(0.0, v.x), max(0.0, v.y), max(0.0, v.z));
}

float3 rotate_y(float3 p, float radians_angle)
{
    float c = cos(radians_angle);
    float s = sin(radians_angle);
    return float3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

float get_height_modulation(float z0, float height)
{
    float h = abs(height / z0);
    if (h > 2.0) return 0.0;
    float val = 1.0 / ((exp(h) + exp(-h)) / 2.0);
    return val * val;
}

float get_radius(float3 p, out float3 P, out float dott)
{
    dott = p.y;
    P = float3(p.x, 0.0, p.z);
    return length(P) / U.axis.x;
}

float find_difference(float t1, float t2)
{
    float v1 = abs(t1 - t2);
    float v2 = abs(t1 - t2 - 2.0 * PI);
    float v3 = abs(t1 - t2 + 2.0 * PI);
    float v4 = abs(t1 - t2 - 4.0 * PI);
    float v5 = abs(t1 - t2 + 4.0 * PI);
    return min(v1, min(v2, min(v3, min(v4, v5))));
}

float get_winding(float rad)
{
    float r = rad + 0.05;
    float t = atan(exp(-0.25 / (0.5 * r)) / U.winding_b) * 2.0 * U.winding_n;
    return t;
}

float get_arm(float rad, float3 p, float disp, float delta, float arm)
{
    float m_workWinding = get_winding(rad);
    float m_workTheta = -atan2(p.x, p.z) - delta;
    float v = abs(find_difference(m_workWinding, m_workTheta + disp)) / PI;
    return pow(clamp(1.0 - v, 0.0, 1.0), arm * 15.0);
}

float calculate_arm_value(float rad, float3 P, float delta, float arm)
{
    int nArms = int(U.no_arms + 0.5);
    float v1 = get_arm(rad, P, U.arm1, delta, arm);
    if (nArms == 1) return v1;
    float v = max(v1, get_arm(rad, P, U.arm2, delta, arm));
    if (nArms == 2) return v;
    v = max(v, get_arm(rad, P, U.arm3, delta, arm));
    if (nArms == 3) return v;
    return max(v, get_arm(rad, P, U.arm4, delta, arm));
}

float get_radial_intensity(float rad, float r0)
{
    float r = exp(-rad / (r0 * 0.5));
    return clamp(r - 0.01, 0.0, 1.0);
}

float3 coord2ray(uint x, uint y)
{
    float xx = float(x) / (float(U.size) * 0.5) - 1.0;
    float yy = float(y) / (float(U.size) * 0.5) - 1.0;
    float4 screen = float4(xx, -yy, 1.0, 1.0);
    float4 world = mul(U.inv_vp, screen);
    return normalize(world.xyz);
}

void accumulate_component(
    int cid,
    float4 p0,
    float4 p1,
    float4 p2,
    float3 spec,
    inout float3 I,
    float3 p,
    float weight,
    float scale_const)
{
    float strength = p0.x;
    float arm = p0.y;
    float r0 = p0.w;

    float active = p1.x;
    float delta = p1.y;
    float winding_mul = p1.z;
    float scale = p1.w;

    float noise_offset = p2.x;
    float noise_tilt = p2.y;
    float ks = p2.z;
    float inner = p2.w;

    if (active != 1.0) return;

    float3 P;
    float dott;
    float radius = get_radius(p, P, dott);
    float z = get_height_modulation(p0.z, dott);

    if (cid == 0)
    {
        float rho_0 = strength * weight;
        float rad = (length(p) + 0.01) * r0;
        rad += 0.01;
        float i = rho_0 * (pow(rad, -0.855) * exp(-pow(rad, 0.25)) - 0.05);
        if (i < 0.0) i = 0.0;
        I += i * spec * scale_const;
        return;
    }

    if (z <= 0.01) return;

    float intensity = get_radial_intensity(radius, r0);
    if (intensity > 0.1) intensity = 0.1;
    if (intensity <= 0.001) return;

    float scale_inner = pow(smoothstep(0.0, 1.0 * inner, radius), 4.0);
    float armVal = 1.0;
    float local_winding = 0.0;
    if (arm != 0.0)
    {
        armVal = calculate_arm_value(radius, P, delta, arm);
        if (winding_mul != 0.0)
            local_winding = get_winding(radius) * winding_mul;
    }
    float val = strength * scale_inner * armVal * z * intensity;
    if (val * weight <= 0.0005) return;

    if (cid == 1)
    {
        float ang = local_winding * PI;
        float3 pd = rotate_y(p, ang);
        float p2v = abs(octave_noise_3d(10.0, ks, scale * 0.1, pd.x, pd.y, pd.z));
        p2v = max(p2v, 0.01);
        p2v = pow(p2v, noise_tilt);
        p2v += noise_offset;
        if (p2v < 0.0) return;
        I += (val * weight) * p2v * spec * scale_const;
    }
    else if (cid == 2)
    {
        float ang = local_winding * PI;
        float3 r = rotate_y(p, ang);
        float p2v = octave_noise_3d(9.0, ks, scale * 0.1, r.x, r.y, r.z);
        p2v = max(p2v - noise_offset, 0.0);
        p2v = clamp(pow(5.0 * p2v, noise_tilt), -10.0, 10.0);
        float s = 0.01;
        I.x = I.x * exp(-p2v * (val * weight) * spec.x * s);
        I.y = I.y * exp(-p2v * (val * weight) * spec.y * s);
        I.z = I.z * exp(-p2v * (val * weight) * spec.z * s);
    }
    else if (cid == 3)
    {
        float ang = local_winding * PI;
        float3 r = rotate_y(p, ang);
        float p2v = get_ridged_mf(r * scale, ks, 9, 2.5, noise_offset, noise_tilt);
        p2v = max(p2v, 0.0);
        float s = 0.01;
        I.x = I.x * exp(-p2v * (val * weight) * spec.x * s);
        I.y = I.y * exp(-p2v * (val * weight) * spec.y * s);
        I.z = I.z * exp(-p2v * (val * weight) * spec.z * s);
    }
    else if (cid == 4)
    {
        float ang = local_winding * PI;
        float3 r = rotate_y(p, ang);
        float p2v = get_ridged_mf(r * scale, ks, 9, 2.5, noise_offset, noise_tilt);
        p2v = max(p2v, 0.0);
        I += (val * weight) * p2v * spec * scale_const;
    }
    else if (cid == 5)
    {
        float perlinnoise = abs(octave_noise_3d(10.0, ks, scale, p.x, p.y, p.z));
        float addNoise = 0.0;
        if (noise_offset != 0.0)
        {
            float ang1 = local_winding * PI;
            float3 r1 = rotate_y(p, ang1);
            addNoise = noise_offset * octave_noise_3d(4.0, -2.0, 0.2, r1.x, r1.y, r1.z);
            float ang2 = (local_winding * 0.5) * PI;
            float3 r2 = rotate_y(p, ang2);
            addNoise += 0.5 * noise_offset * octave_noise_3d(4.0, -2.0, 0.4, r2.x, r2.y, r2.z);
        }
        float v = abs(pow(perlinnoise + 1.0 + addNoise, noise_tilt));
        I += (val * weight) * v * spec * scale_const;
    }
}

float3 post_process(float3 val)
{
    float3 v = val;
    v *= 1.0 / U.exposure;
    v.x = pow(v.x, U.gamma);
    v.y = pow(v.y, U.gamma);
    v.z = pow(v.z, U.gamma);

    float center = (v.x + v.y + v.z) / 3.0;
    float3 tmp = float3(center - v.x, center - v.y, center - v.z);
    v.x = center - U.saturation * tmp.x;
    v.y = center - U.saturation * tmp.y;
    v.z = center - U.saturation * tmp.z;

    return float3(
        clamp(v.x * 10.0, 0.0, 255.0),
        clamp(v.y * 10.0, 0.0, 255.0),
        clamp(v.z * 10.0, 0.0, 255.0));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= U.size || tid.y >= U.size) return;

    uint idx = tid.x + tid.y * U.size;

    float3 dir = coord2ray(tid.x, tid.y);

    float3 camera = U.camera;
    float3 isp1, isp2;
    float t1, t2;
    bool intersects = intersect_sphere(camera, dir, U.axis, isp1, isp2, t1, t2);

    if (t2 > 0.0)
    {
        isp2 = camera;
    }
    if (t1 > 0.0 && t2 > 0.0)
    {
        intersects = false;
    }

    float3 I = float3(0,0,0);

    if (intersects)
    {
        float3 origin = isp1;
        float lengthv = length(isp1 - isp2);
        float3 step_dir = normalize(isp1 - isp2);
        float step = U.ray_step;
        float3 p = origin;

        float3 ll = normalize(isp2 - origin);
        while (dot(p - origin, ll) < lengthv + step)
        {
            step = clamp(pow(length(p - camera), 1.0) * U.ray_step, 0.001, 0.01);

            for (uint ci = 0; ci < U.component_count; ++ci)
            {
                float4 p0 = U.comp_p0[ci];
                float4 p1 = U.comp_p1[ci];
                float4 p2 = U.comp_p2[ci];
                float3 spec = U.comp_spec[ci].xyz;
                accumulate_component(U.class_id[ci], p0, p1, p2, spec, I, p, step * 200.0, U.ray_step);
            }

            p -= step_dir * step;
            I = floor_vec(I);
        }
    }

    I *= 0.01 / U.ray_step;
    float3 c = post_process(I);

    uint R = uint(c.x);
    uint G = uint(c.y);
    uint B = uint(c.z);
    out_pixels[idx] = (R << 16) | (G << 8) | B;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main_rt(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= U.size || tid.y >= U.size) return;

    float3 dir = coord2ray(tid.x, tid.y);

    float3 camera = U.camera;
    float3 isp1, isp2;
    float t1, t2;
    bool intersects = intersect_sphere(camera, dir, U.axis, isp1, isp2, t1, t2);

    if (t2 > 0.0)
    {
        isp2 = camera;
    }
    if (t1 > 0.0 && t2 > 0.0)
    {
        intersects = false;
    }

    float3 I = float3(0, 0, 0);

    if (intersects)
    {
        float3 origin = isp1;
        float lengthv = length(isp1 - isp2);
        float3 step_dir = normalize(isp1 - isp2);
        float step = U.ray_step;
        float3 p = origin;

        float3 ll = normalize(isp2 - origin);
        while (dot(p - origin, ll) < lengthv + step)
        {
            step = clamp(pow(length(p - camera), 1.0) * U.ray_step, 0.001, 0.01);

            for (uint ci = 0; ci < U.component_count; ++ci)
            {
                float4 p0 = U.comp_p0[ci];
                float4 p1 = U.comp_p1[ci];
                float4 p2 = U.comp_p2[ci];
                float3 spec = U.comp_spec[ci].xyz;
                accumulate_component(U.class_id[ci], p0, p1, p2, spec, I, p, step * 200.0, U.ray_step);
            }

            p -= step_dir * step;
            I = floor_vec(I);
        }
    }

    I *= 0.01 / U.ray_step;
    float3 c = post_process(I) / 255.0;
    out_image[tid.xy] = float4(c, 1.0);
}
