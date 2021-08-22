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
 * A mutable single-precision complex number.
 */
public interface Zf {

    float re();

    float im();

    void setRe(float re);

    void setIm(float im);

    void set(float re, float im);

    Zf add(Zf that);

    Zf sub(Zf that);

    Zf conj();

    Zf neg();

    /**
     * Compute the argument of this complex number. The argument is the angle
     * phi between the positive real axis and the point representing this number
     * in the complex plane. The value returned is between -PI (not inclusive)
     * and PI (inclusive), with negative values returned for numbers with
     * negative imaginary parts. Also known as {@code phase} or {@code angle}.
     * 
     * @return the angle of this complex number
     */
    float arg();

    /**
     * Also known as {@code magnitude} or {@code modulus}.
     * 
     * @return the absolute value of this complex number
     */
    float abs();

    /**
     * Multiplicative inverse: {@code 1.0f / this}.
     * 
     * @return the multiplicative inverse of this complex number
     */
    Zf inv();

    Zf mul(Zf that);

    Zf div(Zf that);

    /**
     * Multiplication with a real number.
     * 
     * @param alpha
     * @return the result of the multiplication
     */
    Zf scale(float alpha);

    /**
     * Natural logarithm: {@code ln(this)}.
     * 
     * @return the natural log of this complex number
     */
    Zf ln();

    /**
     * Exponential function: {@code e}<sup>{@code this}</sup>.
     * 
     * @return {@code e} to the power of this complex number
     */
    Zf exp();

    /**
     * Power function of this complex base with a real exponent.
     * <p>
     * Computes {@code this}<sup>{@code exponent}</sup>
     * 
     * @param exponent
     *            real exponent
     * @return this complex number to the power of the real exponent
     */
    Zf pow(float exponent);

    /**
     * Power function of this complex base with a complex exponent.
     * <p>
     * Computes {@code this}<sup>{@code exponent}</sup>
     * 
     * @param exponent
     *            complex exponent
     * @return this complex number to the power of the complex exponent
     */
    Zf pow(Zf exponent);

    boolean isReal();

    boolean isNan();

    boolean isInfinite();

    Zf copy();

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
