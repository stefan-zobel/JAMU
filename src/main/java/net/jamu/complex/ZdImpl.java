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
 * Mutable {@link Zd} implementation.
 */
public final class ZdImpl implements Zd {

    private double re;
    private double im;

    public double re() {
        return re;
    }

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

    public void setRe(double re) {
        this.re = re;
    }

    public void setIm(double im) {
        this.im = im;
    }

    public Zd copy() {
        return new ZdImpl(re, im);
    }

    public Zd add(Zd that) {
        re += that.re();
        im += that.im();
        return this;
    }

    public Zd sub(Zd that) {
        re -= that.re();
        im -= that.im();
        return this;
    }

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
        double scale = re * re + im * im;
        re = re / scale;
        im = -im / scale;
        return this;
    }

    public Zd ln() {
        double abs = abs();
        double phi = arg();
        re = Math.log(abs);
        im = phi;
        return this;
    }

    public Zd exp() {
        double expRe = Math.exp(re);
        double im_ = im;
        re = expRe * Math.cos(im_);
        im = expRe * Math.sin(im_);
        return this;
    }

    public Zd pow(double exponent) {
        return ln().scale(exponent).exp();
    }

    public Zd pow(Zd exponent) {
        return ln().mul(exponent).exp();
    }

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

    public Zd conj() {
        im = -im;
        return this;
    }

    public Zd neg() {
        re = -re;
        im = -im;
        return this;
    }

    // the following methods could also be used for an immutable complex given
    // re() and im()

    public final boolean isReal() {
        return im() == 0.0;
    }

    public final double arg() {
        return Math.atan2(im(), re());
    }

    public final double abs() {
        if (isInfinite()) {
            return Double.POSITIVE_INFINITY;
        }
        // sqrt(a^2 + b^2) without under/overflow
        double re = re();
        double im = im();
        if (Math.abs(re) > Math.abs(im)) {
            double abs = im / re;
            return Math.abs(re) * Math.sqrt(1.0 + abs * abs);
        } else if (im != 0.0) {
            double abs = re / im;
            return Math.abs(im) * Math.sqrt(1.0 + abs * abs);
        }
        return 0.0;
    }

    public final boolean isNan() {
        return Double.isNaN(re()) || Double.isNaN(im());
    }

    public final boolean isInfinite() {
        return Double.isInfinite(re()) || Double.isInfinite(im());
    }

    public final String toString() {
        return re() + "  " + im() + "i";
    }

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
}
