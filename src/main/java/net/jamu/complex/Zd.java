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
 * A mutable double-precision complex number.
 */
public interface Zd {

    double re();

    double im();

    void setRe(double re);

    void setIm(double im);

    void set(double re, double im);

    Zd add(Zd that);

    Zd sub(Zd that);

    Zd conj();

    Zd neg();

    /**
     * Compute the argument of this complex number. The argument is the angle
     * phi between the positive real axis and the point representing this number
     * in the complex plane. The value returned is between -PI (not inclusive)
     * and PI (inclusive), with negative values returned for numbers with
     * negative imaginary parts. Also known as {@code phase} or {@code angle}.
     * 
     * @return the angle of this complex number
     */
    double arg();

    /**
     * Also known as {@code magnitude} or {@code modulus}.
     * 
     * @return the absolute value of this complex number
     */
    double abs();

    /**
     * Multiplicative inverse: {@code 1.0 / this}.
     * 
     * @return the multiplicative inverse of this complex number
     */
    Zd inv();

    Zd mul(Zd that);

    Zd div(Zd that);

    /**
     * Multiplication with a real number.
     * 
     * @param alpha
     * @return the result of the multiplication
     */
    Zd scale(double alpha);

    /**
     * Natural logarithm: {@code ln(this)}.
     * 
     * @return the natural log of this complex number
     */
    Zd ln();

    /**
     * Exponential function: {@code e}<sup>{@code this}</sup>.
     * 
     * @return {@code e} to the power of this complex number
     */
    Zd exp();

    /**
     * Power function of this complex base with a real exponent.
     * <p>
     * Computes {@code this}<sup>{@code exponent}</sup>
     * 
     * @param exponent
     *            real exponent
     * @return this complex number to the power of the real exponent
     */
    Zd pow(double exponent);

    /**
     * Power function of this complex base with a complex exponent.
     * <p>
     * Computes {@code this}<sup>{@code exponent}</sup>
     * 
     * @param exponent
     *            complex exponent
     * @return this complex number to the power of the complex exponent
     */
    Zd pow(Zd exponent);

    boolean isReal();

    boolean isNan();

    boolean isInfinite();

    Zd copy();

    String toString();

    /**
     * Get this complex number as a formatted string.
     * 
     * @param format
     *            a format string in
     *            {@link java.util.Formatter#format(String, Object...)} format
     *            string syntax
     * @return this complex number formatted as specified in the {@code format}
     *         syntax
     * @since 1.3
     */
    String toString(String format);
}
