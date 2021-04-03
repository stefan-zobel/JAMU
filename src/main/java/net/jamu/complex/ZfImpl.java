/*
 * Copyright 2018 Stefan Zobel
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

    public float re() {
        return re;
    }

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

    public void setRe(float re) {
        this.re = re;
    }

    public void setIm(float im) {
        this.im = im;
    }

    public void set(float re, float im) {
        this.re = re;
        this.im = im;
    }

    public Zf copy() {
        return new ZfImpl(re, im);
    }

    public Zf add(Zf that) {
        re += that.re();
        im += that.im();
        return this;
    }

    public Zf sub(Zf that) {
        re -= that.re();
        im -= that.im();
        return this;
    }

    public Zf mul(Zf that) {
        if (isInfinite() || that.isInfinite()) {
            re = Float.POSITIVE_INFINITY;
            im = Float.POSITIVE_INFINITY;
            return this;
        }
        float this_re = re;
        float that_re = that.re();
        re = this_re * that_re - im * that.im();
        im = im * that_re + this_re * that.im();
        return this;
    }

    public Zf div(Zf that) {
        float c = that.re();
        float d = that.im();
        if (c == 0.0f && d == 0.0f) {
            re = Float.NaN;
            im = Float.NaN;
            return this;
        }
        if (that.isInfinite() && !this.isInfinite()) {
            re = 0.0f;
            im = 0.0f;
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
        float scale = re * re + im * im;
        re = re / scale;
        im = -im / scale;
        return this;
    }

    public Zf ln() {
        float abs = abs();
        float phi = arg();
        re = (float) Math.log(abs);
        im = phi;
        return this;
    }

    public Zf exp() {
        double expRe = Math.exp(re);
        float im_ = im;
        re = (float) (expRe * Math.cos(im_));
        im = (float) (expRe * Math.sin(im_));
        return this;
    }

    public Zf pow(float exponent) {
        return ln().scale(exponent).exp();
    }

    public Zf pow(Zf exponent) {
        return ln().mul(exponent).exp();
    }

    public Zf scale(float alpha) {
        if (isInfinite() || Float.isInfinite(alpha)) {
            re = Float.POSITIVE_INFINITY;
            im = Float.POSITIVE_INFINITY;
            return this;
        }
        re = alpha * re;
        im = alpha * im;
        return this;
    }

    public Zf conj() {
        im = -im;
        return this;
    }

    public Zf neg() {
        re = -re;
        im = -im;
        return this;
    }

    // the following methods could also be used for an immutable complex given
    // re() and im()

    public final boolean isReal() {
        return im() == 0.0f;
    }

    public final float arg() {
        return (float) Math.atan2(im(), re());
    }

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

    public final boolean isNan() {
        return Float.isNaN(re()) || Float.isNaN(im());
    }

    public final boolean isInfinite() {
        return Float.isInfinite(re()) || Float.isInfinite(im());
    }

    public final String toString() {
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
        buf.append(String.format("%.6E", re_)).append("  ");
        if (im_ >= 0.0f) {
            buf.append("+");
        }
        buf.append(String.format("%.6E", im_)).append("i");
        return buf.toString();
    }

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

    public final int hashCode() {
        int h = 0x7FFFF + Float.floatToIntBits(re);
        h = ((h << 19) - h) + Float.floatToIntBits(im);
        return (h << 19) - h;
    }
}
