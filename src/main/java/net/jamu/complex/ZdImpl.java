/*
 * Copyright 2018, 2021 Stefan Zobel
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
 * Mutable {@link Zd} implementation.
 */
public final class ZdImpl implements Zd {

    private double re;
    private double im;

    @Override
    public double re() {
        return re;
    }

    @Override
    public double im() {
        return im;
    }

    public ZdImpl(double re) {
        this(re, 0.0);
    }

    public ZdImpl(double re, double im) {
        this.re = re;
        this.im = im;
    }

    public ZdImpl(Zd that) {
        this.re = that.re();
        this.im = that.im();
    }

    public static Zd fromPolar(double radius, double phi) {
        if (radius < 0.0) {
            throw new IllegalArgumentException("radius must be positive : " + radius);
        }
        return new ZdImpl(radius * Math.cos(phi), radius * Math.sin(phi));
    }

    public static double abs(double re, double im) {
        // sqrt(a^2 + b^2) without under/overflow
        if (im == 0.0) {
            return re >= 0.0 ? re : -re;
        } else if (Math.abs(re) > Math.abs(im)) {
            double abs = im / re;
            return Math.abs(re) * Math.sqrt(1.0 + abs * abs);
        } else if (im != 0.0) {
            double abs = re / im;
            return Math.abs(im) * Math.sqrt(1.0 + abs * abs);
        }
        return 0.0;
    }

    @Override
    public void setRe(double re) {
        this.re = re;
    }

    @Override
    public void setIm(double im) {
        this.im = im;
    }

    @Override
    public void set(double re, double im) {
        this.re = re;
        this.im = im;
    }

    @Override
    public Zd copy() {
        return new ZdImpl(re, im);
    }

    @Override
    public Zd add(Zd that) {
        re += that.re();
        im += that.im();
        return this;
    }

    @Override
    public Zd sub(Zd that) {
        re -= that.re();
        im -= that.im();
        return this;
    }

    @Override
    public Zd mul(Zd that) {
        if (isInfinite() || that.isInfinite()) {
            re = Double.POSITIVE_INFINITY;
            im = Double.POSITIVE_INFINITY;
            return this;
        }
        double this_re = re;
        double that_re = that.re();
        re = this_re * that_re - im * that.im();
        im = im * that_re + this_re * that.im();
        return this;
    }

    @Override
    public Zd div(Zd that) {
        double c = that.re();
        double d = that.im();
        if (c == 0.0 && d == 0.0) {
            re = Double.NaN;
            im = Double.NaN;
            return this;
        }
        if (that.isInfinite() && !this.isInfinite()) {
            re = 0.0;
            im = 0.0;
            return this;
        }
        // limit overflow/underflow
        if (Math.abs(c) < Math.abs(d)) {
            double q = c / d;
            double denom = c * q + d;
            double real = re;
            re = (real * q + im) / denom;
            im = (im * q - real) / denom;
        } else {
            double q = d / c;
            double denom = d * q + c;
            double real = re;
            re = (im * q + real) / denom;
            im = (im - real * q) / denom;
        }
        return this;
    }

    @Override
    public Zd inv() {
        if (re == 0.0 && im == 0.0) {
            re = Double.POSITIVE_INFINITY;
            im = Double.POSITIVE_INFINITY;
            return this;
        }
        if (isInfinite()) {
            re = 0.0;
            im = 0.0;
            return this;
        }
        // the scaling from div(), with a numerator of (1, 0)
        double c = re;
        double d = im;
        if (Math.abs(c) < Math.abs(d)) {
            double q = c / d;
            double denom = c * q + d;
            re = q / denom;
            im = -1.0 / denom;
        } else {
            double q = d / c;
            double denom = d * q + c;
            re = 1.0 / denom;
            im = -q / denom;
        }
        return this;
    }

    @Override
    public Zd ln() {
        double abs = abs();
        double phi = arg();
        re = Math.log(abs);
        im = phi;
        return this;
    }

    @Override
    public Zd exp() {
        double c = Math.cos(im);
        double s = Math.sin(im);
        double h = Math.exp(re);
        if (Double.isInfinite(h)) {
            // e^re overflows although the product with cos or sin need not
            h = Math.exp(re / 2.0);
            re = (c == 0.0) ? c : h * c * h;
            im = (s == 0.0) ? s : h * s * h;
        } else {
            re = h * c;
            im = h * s;
        }
        return this;
    }

    @Override
    public Zd pow(double exponent) {
        if (isDegenerate()) {
            return degeneratePow(exponent);
        }
        return ln().scale(exponent).exp();
    }

    @Override
    public Zd pow(Zd exponent) {
        if (isDegenerate()) {
            return degeneratePow(exponent);
        }
        return ln().mul(exponent).exp();
    }

    // a base that ln() cannot carry
    private boolean isDegenerate() {
        return (re == 0.0 && im == 0.0) || isInfinite();
    }

    // the values of Math.pow, mirrored for an infinite base
    private Zd degeneratePow(double exponent) {
        boolean zeroBase = (re == 0.0 && im == 0.0);
        if (isNan() || Double.isNaN(exponent)) {
            set(Double.NaN, Double.NaN);
        } else if (exponent == 0.0) {
            set(1.0, 0.0);
        } else if (zeroBase == (exponent > 0.0)) {
            set(0.0, 0.0);
        } else {
            set(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
        }
        return this;
    }

    private Zd degeneratePow(Zd exponent) {
        double a = exponent.re();
        double b = exponent.im();
        if (a == 0.0 && b == 0.0) {
            set(1.0, 0.0);
            return this;
        }
        if (a == 0.0 || Double.isNaN(b) || Double.isInfinite(b)) {
            // only a real exponent is defined here, and 0^(bi) is not
            set(Double.NaN, Double.NaN);
            return this;
        }
        return degeneratePow(a);
    }

    @Override
    public Zd scale(double alpha) {
        if (isInfinite() || Double.isInfinite(alpha)) {
            re = Double.POSITIVE_INFINITY;
            im = Double.POSITIVE_INFINITY;
            return this;
        }
        re = alpha * re;
        im = alpha * im;
        return this;
    }

    @Override
    public Zd conj() {
        im = -im;
        return this;
    }

    @Override
    public Zd neg() {
        re = -re;
        im = -im;
        return this;
    }

    // the following methods could also be used for an immutable complex given
    // re() and im()

    @Override
    public final boolean isReal() {
        return im() == 0.0;
    }

    @Override
    public final double arg() {
        return Math.atan2(im(), re());
    }

    @Override
    public final double abs() {
        if (isInfinite()) {
            return Double.POSITIVE_INFINITY;
        }
        // sqrt(a^2 + b^2) without under/overflow
        double re = re();
        double im = im();
        if (im == 0.0) {
            return re >= 0.0 ? re : -re;
        } else if (Math.abs(re) > Math.abs(im)) {
            double abs = im / re;
            return Math.abs(re) * Math.sqrt(1.0 + abs * abs);
        } else if (im != 0.0) {
            double abs = re / im;
            return Math.abs(im) * Math.sqrt(1.0 + abs * abs);
        }
        return 0.0;
    }

    @Override
    public final boolean isNan() {
        return Double.isNaN(re()) || Double.isNaN(im());
    }

    @Override
    public final boolean isInfinite() {
        return Double.isInfinite(re()) || Double.isInfinite(im());
    }

    @Override
    public final String toString() {
        return toString("%.10E");
    }

    @Override
    public String toString(String format) {
        double re_ = re();
        double im_ = im();
        // fix negative zero
        if (re_ == 0.0) {
            re_ = 0.0;
        }
        if (im_ == 0.0) {
            im_ = 0.0;
        }
        StringBuilder buf = new StringBuilder(40);
        if (re_ >= 0.0) {
            buf.append("+");
        }
        buf.append(String.format(format, re_)).append("  ");
        if (im_ >= 0.0) {
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
        if (that instanceof Zd) {
            Zd other = (Zd) that;
            if (other.isNan()) {
                return this.isNan();
            }
            return re() == other.re() && im() == other.im();
        }
        return false;
    }

    @Override
    public final int hashCode() {
        long bits = Double.doubleToLongBits(re);
        int h = 0x7FFFF + (int) (bits ^ (bits >>> 32));
        bits = Double.doubleToLongBits(im);
        h = ((h << 19) - h) + (int) (bits ^ (bits >>> 32));
        return (h << 19) - h;
    }
}
