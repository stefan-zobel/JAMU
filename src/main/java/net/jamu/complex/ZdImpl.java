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
        double c = Math.cos(phi);
        double s = Math.sin(phi);
        // an exact zero stays zero even for an infinite radius
        return new ZdImpl(c == 0.0 ? c : radius * c, s == 0.0 ? s : radius * s);
    }

    public static double abs(double re, double im) {
        // sqrt(a^2 + b^2) without under/overflow
        if (im == 0.0) {
            return Math.abs(re);
        } else if (Math.abs(re) > Math.abs(im)) {
            double abs = im / re;
            return Math.abs(re) * Math.sqrt(1.0 + abs * abs);
        } else {
            double abs = re / im;
            return Math.abs(im) * Math.sqrt(1.0 + abs * abs);
        }
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
        double a = re;
        double b = im;
        double c = that.re();
        double d = that.im();
        if (isInfinite() || that.isInfinite()) {
            // C99 Annex G: an infinite operand still fixes the direction
            if (Double.isInfinite(a) || Double.isInfinite(b)) {
                a = Math.copySign(Double.isInfinite(a) ? 1.0 : 0.0, a);
                b = Math.copySign(Double.isInfinite(b) ? 1.0 : 0.0, b);
            }
            if (Double.isInfinite(c) || Double.isInfinite(d)) {
                c = Math.copySign(Double.isInfinite(c) ? 1.0 : 0.0, c);
                d = Math.copySign(Double.isInfinite(d) ? 1.0 : 0.0, d);
            }
            a = zeroIfNan(a);
            b = zeroIfNan(b);
            c = zeroIfNan(c);
            d = zeroIfNan(d);
            if ((a == 0.0 && b == 0.0) || (c == 0.0 && d == 0.0)) {
                // zero times infinity has no direction and no modulus
                re = Double.NaN;
                im = Double.NaN;
                return this;
            }
            re = unbounded(a * c - b * d);
            im = unbounded(a * d + b * c);
            return this;
        }
        re = a * c - b * d;
        im = a * d + b * c;
        return this;
    }

    private static double zeroIfNan(double x) {
        return Double.isNaN(x) ? Math.copySign(0.0, x) : x;
    }

    // C99 scales by infinity here; an exact zero keeps its sign instead
    private static double unbounded(double x) {
        if (x == 0.0 || Double.isNaN(x)) {
            return x;
        }
        return (x > 0.0) ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
    }

    @Override
    public Zd div(Zd that) {
        double c = that.re();
        double d = that.im();
        if (c == 0.0 && d == 0.0) {
            // zero over zero has no value, anything else over zero is inv(0)
            if (re == 0.0 && im == 0.0) {
                re = Double.NaN;
                im = Double.NaN;
            } else {
                re = Double.POSITIVE_INFINITY;
                im = Double.POSITIVE_INFINITY;
            }
            return this;
        }
        boolean thisInfinite = isInfinite();
        if (that.isInfinite() && !thisInfinite) {
            re = 0.0;
            im = 0.0;
            return this;
        }
        if (thisInfinite && Double.isFinite(c) && Double.isFinite(d)) {
            // C99 Annex G: an infinite numerator still fixes the direction
            double a = Math.copySign(Double.isInfinite(re) ? 1.0 : 0.0, re);
            double b = Math.copySign(Double.isInfinite(im) ? 1.0 : 0.0, im);
            re = unbounded(a * c + b * d);
            im = unbounded(b * c - a * d);
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
            re = (c == 0.0) ? c : h * c;
            im = (s == 0.0) ? s : h * s;
        }
        return this;
    }

    @Override
    public Zd sqrt() {
        if (Double.isInfinite(im)) {
            // C99: an infinite imaginary part decides, whatever the real part is
            re = Double.POSITIVE_INFINITY;
            im = Math.copySign(Double.POSITIVE_INFINITY, im);
            return this;
        }
        if (re == 0.0 && im == 0.0) {
            re = 0.0;
            return this;
        }
        // Kahan: t is built from |re| so that nothing cancels
        double b = im;
        double t = Math.sqrt((Math.abs(re) + abs()) / 2.0);
        if (re >= 0.0) {
            re = t;
            im = b / (2.0 * t);
        } else {
            re = Math.abs(b) / (2.0 * t);
            im = Math.copySign(t, b);
        }
        return this;
    }

    @Override
    public Zd pow(double exponent) {
        if (isDegenerate()) {
            return degeneratePow(exponent);
        }
        if (Double.isInfinite(exponent)) {
            return infinitePow(exponent);
        }
        return ln().scale(exponent).exp();
    }

    @Override
    public Zd pow(Zd exponent) {
        if (isDegenerate()) {
            return degeneratePow(exponent);
        }
        if (exponent.isInfinite()) {
            if (exponent.im() == 0.0) {
                return infinitePow(exponent.re());
            }
            set(Double.NaN, Double.NaN);
            return this;
        }
        return ln().mul(exponent).exp();
    }

    // the values of Math.pow, with the modulus in place of |x|
    private Zd infinitePow(double exponent) {
        double r = abs();
        if (Double.isNaN(r) || r == 1.0) {
            set(Double.NaN, Double.NaN);
        } else if ((r > 1.0) == (exponent > 0.0)) {
            set(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
        } else {
            set(0.0, 0.0);
        }
        return this;
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
        double a = re;
        double b = im;
        double c = alpha;
        if (isInfinite() || Double.isInfinite(alpha)) {
            if (Double.isInfinite(a) || Double.isInfinite(b)) {
                a = Math.copySign(Double.isInfinite(a) ? 1.0 : 0.0, a);
                b = Math.copySign(Double.isInfinite(b) ? 1.0 : 0.0, b);
            }
            if (Double.isInfinite(c)) {
                c = Math.copySign(1.0, c);
            }
            a = zeroIfNan(a);
            b = zeroIfNan(b);
            c = zeroIfNan(c);
            if ((a == 0.0 && b == 0.0) || c == 0.0) {
                re = Double.NaN;
                im = Double.NaN;
                return this;
            }
            re = unbounded(a * c);
            im = unbounded(b * c);
            return this;
        }
        re = alpha * a;
        im = alpha * b;
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
        return im() == 0.0 && !Double.isNaN(re());
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
            return Math.abs(re);
        } else if (Math.abs(re) > Math.abs(im)) {
            double abs = im / re;
            return Math.abs(re) * Math.sqrt(1.0 + abs * abs);
        } else {
            double abs = re / im;
            return Math.abs(im) * Math.sqrt(1.0 + abs * abs);
        }
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
        // equals() sees one value in every NaN and does not tell the zeros apart
        boolean nan = isNan();
        long bits = Double.doubleToLongBits(nan ? Double.NaN : re + 0.0);
        int h = 0x7FFFF + (int) (bits ^ (bits >>> 32));
        bits = Double.doubleToLongBits(nan ? Double.NaN : im + 0.0);
        h = ((h << 19) - h) + (int) (bits ^ (bits >>> 32));
        return (h << 19) - h;
    }
}
