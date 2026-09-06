/*
 * Copyright 2018, 2026 Stefan Zobel
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package net.jamu.complex;

/**
 * Mutable {@link Zf} implementation.
 */
public final class ZfImpl implements Zf {

    private float re;
    private float im;

    @Override
    public float re() {
        return re;
    }

    @Override
    public float im() {
        return im;
    }

    public ZfImpl(float re) {
        this(re, 0.0f);
    }

    public ZfImpl(float re, float im) {
        this.re = re;
        this.im = im;
    }

    public ZfImpl(Zf that) {
        this.re = that.re();
        this.im = that.im();
    }

    public static Zf fromPolar(float radius, float phi) {
        if (radius < 0.0f) {
            throw new IllegalArgumentException("radius must be positive : " + radius);
        }
        return new ZfImpl((float) (radius * Math.cos(phi)), (float) (radius * Math.sin(phi)));
    }

    public static float abs(float re, float im) {
        // sqrt(a^2 + b^2) without under/overflow
        if (im == 0.0f) {
            return re >= 0.0f ? re : -re;
        } else if (Math.abs(re) > Math.abs(im)) {
            double abs = im / re;
            return (float) (Math.abs(re) * Math.sqrt(1.0 + abs * abs));
        } else if (im != 0.0f) {
            double abs = re / im;
            return (float) (Math.abs(im) * Math.sqrt(1.0 + abs * abs));
        }
        return 0.0f;
    }

    @Override
    public void setRe(float re) {
        this.re = re;
    }

    @Override
    public void setIm(float im) {
        this.im = im;
    }

    @Override
    public void set(float re, float im) {
        this.re = re;
        this.im = im;
    }

    @Override
    public Zf copy() {
        return new ZfImpl(re, im);
    }

    @Override
    public Zf add(Zf that) {
        re += that.re();
        im += that.im();
        return this;
    }

    @Override
    public Zf sub(Zf that) {
        re -= that.re();
        im -= that.im();
        return this;
    }

    @Override
    public Zf mul(Zf that) {
        float a = re;
        float b = im;
        float c = that.re();
        float d = that.im();
        if (isInfinite() || that.isInfinite()) {
            boolean thisZero = (a == 0.0f && b == 0.0f);
            boolean thatZero = (c == 0.0f && d == 0.0f);
            if (thisZero || thatZero) {
                // zero times infinity has no direction and no modulus
                re = Float.NaN;
                im = Float.NaN;
                return this;
            }
            // C99 Annex G: an infinite operand still fixes the direction
            if (Float.isInfinite(a) || Float.isInfinite(b)) {
                a = Math.copySign(Float.isInfinite(a) ? 1.0f : 0.0f, a);
                b = Math.copySign(Float.isInfinite(b) ? 1.0f : 0.0f, b);
            }
            if (Float.isInfinite(c) || Float.isInfinite(d)) {
                c = Math.copySign(Float.isInfinite(c) ? 1.0f : 0.0f, c);
                d = Math.copySign(Float.isInfinite(d) ? 1.0f : 0.0f, d);
            }
            a = zeroIfNan(a);
            b = zeroIfNan(b);
            c = zeroIfNan(c);
            d = zeroIfNan(d);
            re = unbounded(a * c - b * d);
            im = unbounded(a * d + b * c);
            return this;
        }
        re = a * c - b * d;
        im = a * d + b * c;
        return this;
    }

    private static float zeroIfNan(float x) {
        return Float.isNaN(x) ? Math.copySign(0.0f, x) : x;
    }

    // C99 scales by infinity here; an exact zero keeps its sign instead
    private static float unbounded(float x) {
        if (x == 0.0f || Float.isNaN(x)) {
            return x;
        }
        return (x > 0.0f) ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
    }

    @Override
    public Zf div(Zf that) {
        float c = that.re();
        float d = that.im();
        if (c == 0.0f && d == 0.0f) {
            // zero over zero has no value, anything else over zero is inv(0)
            if (re == 0.0f && im == 0.0f) {
                re = Float.NaN;
                im = Float.NaN;
            } else {
                re = Float.POSITIVE_INFINITY;
                im = Float.POSITIVE_INFINITY;
            }
            return this;
        }
        boolean thisInfinite = isInfinite();
        if (that.isInfinite() && !thisInfinite) {
            re = 0.0f;
            im = 0.0f;
            return this;
        }
        if (thisInfinite && Float.isFinite(c) && Float.isFinite(d)) {
            // C99 Annex G: an infinite numerator still fixes the direction
            float a = Math.copySign(Float.isInfinite(re) ? 1.0f : 0.0f, re);
            float b = Math.copySign(Float.isInfinite(im) ? 1.0f : 0.0f, im);
            re = unbounded(a * c + b * d);
            im = unbounded(b * c - a * d);
            return this;
        }
        // limit overflow/underflow
        if (Math.abs(c) < Math.abs(d)) {
            float q = c / d;
            float denom = c * q + d;
            float real = re;
            re = (real * q + im) / denom;
            im = (im * q - real) / denom;
        } else {
            float q = d / c;
            float denom = d * q + c;
            float real = re;
            re = (im * q + real) / denom;
            im = (im - real * q) / denom;
        }
        return this;
    }

    @Override
    public Zf inv() {
        if (re == 0.0f && im == 0.0f) {
            re = Float.POSITIVE_INFINITY;
            im = Float.POSITIVE_INFINITY;
            return this;
        }
        if (isInfinite()) {
            re = 0.0f;
            im = 0.0f;
            return this;
        }
        // the scaling from div(), with a numerator of (1, 0)
        float c = re;
        float d = im;
        if (Math.abs(c) < Math.abs(d)) {
            float q = c / d;
            float denom = c * q + d;
            re = q / denom;
            im = -1.0f / denom;
        } else {
            float q = d / c;
            float denom = d * q + c;
            re = 1.0f / denom;
            im = -q / denom;
        }
        return this;
    }

    @Override
    public Zf ln() {
        float abs = abs();
        float phi = arg();
        re = (float) Math.log(abs);
        im = phi;
        return this;
    }

    @Override
    public Zf exp() {
        double expRe = Math.exp(re);
        double c = Math.cos(im);
        double s = Math.sin(im);
        // an exact zero stays zero even when expRe has overflown
        re = (c == 0.0) ? (float) c : (float) (expRe * c);
        im = (s == 0.0) ? (float) s : (float) (expRe * s);
        return this;
    }

    @Override
    public Zf pow(float exponent) {
        if (isDegenerate()) {
            return degeneratePow(exponent);
        }
        if (Float.isInfinite(exponent)) {
            return infinitePow(exponent);
        }
        return ln().scale(exponent).exp();
    }

    @Override
    public Zf pow(Zf exponent) {
        if (isDegenerate()) {
            return degeneratePow(exponent);
        }
        if (exponent.isInfinite()) {
            if (exponent.im() == 0.0f) {
                return infinitePow(exponent.re());
            }
            set(Float.NaN, Float.NaN);
            return this;
        }
        return ln().mul(exponent).exp();
    }

    // the values of Math.pow, with the modulus in place of |x|
    private Zf infinitePow(float exponent) {
        float r = abs();
        if (Float.isNaN(r) || r == 1.0f) {
            set(Float.NaN, Float.NaN);
        } else if ((r > 1.0f) == (exponent > 0.0f)) {
            set(Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY);
        } else {
            set(0.0f, 0.0f);
        }
        return this;
    }

    // a base that ln() cannot carry
    private boolean isDegenerate() {
        return (re == 0.0f && im == 0.0f) || isInfinite();
    }

    // the values of Math.pow, mirrored for an infinite base
    private Zf degeneratePow(float exponent) {
        boolean zeroBase = (re == 0.0f && im == 0.0f);
        if (isNan() || Float.isNaN(exponent)) {
            set(Float.NaN, Float.NaN);
        } else if (exponent == 0.0f) {
            set(1.0f, 0.0f);
        } else if (zeroBase == (exponent > 0.0f)) {
            set(0.0f, 0.0f);
        } else {
            set(Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY);
        }
        return this;
    }

    private Zf degeneratePow(Zf exponent) {
        float a = exponent.re();
        float b = exponent.im();
        if (a == 0.0f && b == 0.0f) {
            set(1.0f, 0.0f);
            return this;
        }
        if (a == 0.0f || Float.isNaN(b) || Float.isInfinite(b)) {
            // only a real exponent is defined here, and 0^(bi) is not
            set(Float.NaN, Float.NaN);
            return this;
        }
        return degeneratePow(a);
    }

    @Override
    public Zf scale(float alpha) {
        float a = re;
        float b = im;
        float c = alpha;
        if (isInfinite() || Float.isInfinite(alpha)) {
            if ((a == 0.0f && b == 0.0f) || c == 0.0f) {
                re = Float.NaN;
                im = Float.NaN;
                return this;
            }
            if (Float.isInfinite(a) || Float.isInfinite(b)) {
                a = Math.copySign(Float.isInfinite(a) ? 1.0f : 0.0f, a);
                b = Math.copySign(Float.isInfinite(b) ? 1.0f : 0.0f, b);
            }
            if (Float.isInfinite(c)) {
                c = Math.copySign(1.0f, c);
            }
            re = unbounded(zeroIfNan(a) * zeroIfNan(c));
            im = unbounded(zeroIfNan(b) * zeroIfNan(c));
            return this;
        }
        re = alpha * a;
        im = alpha * b;
        return this;
    }

    @Override
    public Zf conj() {
        im = -im;
        return this;
    }

    @Override
    public Zf neg() {
        re = -re;
        im = -im;
        return this;
    }

    // the following methods could also be used for an immutable complex given
    // re() and im()

    @Override
    public final boolean isReal() {
        return im() == 0.0f;
    }

    @Override
    public final float arg() {
        return (float) Math.atan2(im(), re());
    }

    @Override
    public final float abs() {
        if (isInfinite()) {
            return Float.POSITIVE_INFINITY;
        }
        // sqrt(a^2 + b^2) without under/overflow
        float re = re();
        float im = im();
        if (im == 0.0f) {
            return re >= 0.0f ? re : -re;
        } else if (Math.abs(re) > Math.abs(im)) {
            double abs = im / re;
            return (float) (Math.abs(re) * Math.sqrt(1.0 + abs * abs));
        } else if (im != 0.0f) {
            double abs = re / im;
            return (float) (Math.abs(im) * Math.sqrt(1.0 + abs * abs));
        }
        return 0.0f;
    }

    @Override
    public final boolean isNan() {
        return Float.isNaN(re()) || Float.isNaN(im());
    }

    @Override
    public final boolean isInfinite() {
        return Float.isInfinite(re()) || Float.isInfinite(im());
    }

    @Override
    public final String toString() {
        return toString("%.6E");
    }

    @Override
    public String toString(String format) {
        float re_ = re();
        float im_ = im();
        // fix negative zero
        if (re_ == 0.0f) {
            re_ = 0.0f;
        }
        if (im_ == 0.0f) {
            im_ = 0.0f;
        }
        StringBuilder buf = new StringBuilder(40);
        if (re_ >= 0.0f) {
            buf.append("+");
        }
        buf.append(String.format(format, re_)).append("  ");
        if (im_ >= 0.0f) {
            buf.append("+");
        }
        buf.append(String.format(format, im_)).append("i");
        return buf.toString();
    }

    @Override
    public final boolean equals(Object that) {
        if (this == that) {
            return true;
        }
        if (that instanceof Zf) {
            Zf other = (Zf) that;
            if (other.isNan()) {
                return this.isNan();
            }
            return re() == other.re() && im() == other.im();
        }
        return false;
    }

    @Override
    public final int hashCode() {
        int h = 0x7FFFF + Float.floatToIntBits(re);
        h = ((h << 19) - h) + Float.floatToIntBits(im);
        return (h << 19) - h;
    }
}
