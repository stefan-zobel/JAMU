/*
 * Copyright 2026 Stefan Zobel
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
package net.jamu.matrix;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Test;

/**
 * Tests for the absolute and the relative zeroize methods.
 */
public final class ZeroizeTest {

    private static final double[] SCALES = { 1.0e0, 1.0e-10, 1.0e-15, 1.0e-16, 1.0e-20, 1.0e20 };
    // every seventh entry is set to 1e-18 of the scale, so it is relative noise
    private static final int NOISE_STRIDE = 7;
    private static final int N = 40;

    @Test
    public void testTheRelativeFormIsScaleInvariant() {
        int expected = survivors(noisy(1.0));
        assertTrue("the fixture must contain noise", expected < N * N);
        for (double s : SCALES) {
            String at = " at scale " + s;
            assertEquals("MatrixD" + at, expected,
                    nonZeroD(noisy(s).zeroizeSubEpsilonRelativeInplace(1)));
            assertEquals("MatrixF" + at, expected,
                    nonZeroF(noisyF(s).zeroizeSubEpsilonRelativeInplace(1)));
            assertEquals("ComplexMatrixD" + at, expected,
                    nonZeroComplexD(noisyComplex(s).zeroizeSubEpsilonRelativeInplace(1)));
            assertEquals("ComplexMatrixF" + at, expected,
                    nonZeroComplexF(noisyComplexF(s).zeroizeSubEpsilonRelativeInplace(1)));
        }
    }

    @Test
    public void testTheAbsoluteFormIsNotScaleInvariant() {
        // pins today's behaviour of the sibling, which this slice must not move
        assertEquals("at scale 1", N * N, nonZeroD(noiseless(1.0e0).zeroizeSubEpsilonInplace(1)));
        assertEquals("at scale 1e-15", N * N, nonZeroD(noiseless(1.0e-15).zeroizeSubEpsilonInplace(1)));
        assertEquals("at scale 1e-16", 0, nonZeroD(noiseless(1.0e-16).zeroizeSubEpsilonInplace(1)));
        assertEquals("at scale 1e-20", 0, nonZeroD(noiseless(1.0e-20).zeroizeSubEpsilonInplace(1)));
        // and at a large scale it removes nothing, not even the noise
        assertEquals("at scale 1e20", N * N, nonZeroD(noisy(1.0e20).zeroizeSubEpsilonInplace(1)));
    }

    @Test
    public void testTheLargestEntrySurvives() {
        for (double s : SCALES) {
            String at = " at scale " + s;
            MatrixD d = noisy(s);
            double max = maxAbsD(d);
            d.zeroizeSubEpsilonRelativeInplace(1);
            assertEquals("MatrixD" + at, max, maxAbsD(d), 0.0);
            ComplexMatrixD cd = noisyComplex(s);
            double cmax = maxAbsComplexD(cd);
            cd.zeroizeSubEpsilonRelativeInplace(1);
            assertEquals("ComplexMatrixD" + at, cmax, maxAbsComplexD(cd), 0.0);
        }
    }

    @Test
    public void testNonFiniteEntriesSurviveAndDoNotDriveTheThreshold() {
        MatrixD d = noisy(1.0);
        int expected = survivors(noisy(1.0));
        d.set(0, 0, Double.POSITIVE_INFINITY);
        d.set(1, 0, Double.NaN);
        d.set(2, 0, Double.NEGATIVE_INFINITY);
        d.zeroizeSubEpsilonRelativeInplace(1);
        assertTrue("+Inf survives", d.get(0, 0) == Double.POSITIVE_INFINITY);
        assertTrue("NaN survives", Double.isNaN(d.get(1, 0)));
        assertTrue("-Inf survives", d.get(2, 0) == Double.NEGATIVE_INFINITY);
        // the three replaced cells were finite before, so at most three counts move
        int got = nonZeroD(d);
        assertTrue("the noise still went, got " + got + " for about " + expected,
                Math.abs(got - expected) <= 3);
    }

    @Test
    public void testComplexIsJudgedByModulus() {
        ComplexMatrixD m = Matrices.createComplexD(2, 2);
        m.set(0, 0, 5.0, 0.0);
        m.set(1, 0, 1.0e-20, 5.0);
        m.set(0, 1, 1.0e-20, 3.0e-20);
        m.set(1, 1, 0.0, 5.0);
        m.zeroizeSubEpsilonRelativeInplace(1);
        assertEquals("a tiny real part beside a large imaginary one stays", 1.0e-20,
                m.get(1, 0).re(), 0.0);
        assertEquals("and so does the imaginary part", 5.0, m.get(1, 0).im(), 0.0);
        assertEquals("a tiny modulus falls, real part", 0.0, m.get(0, 1).re(), 0.0);
        assertEquals("a tiny modulus falls, imaginary part", 0.0, m.get(0, 1).im(), 0.0);
        assertEquals("the largest entry stays", 5.0, m.get(0, 0).re(), 0.0);

        ComplexMatrixF f = Matrices.createComplexF(2, 2);
        f.set(0, 0, 5.0f, 0.0f);
        f.set(1, 0, 1.0e-20f, 5.0f);
        f.set(0, 1, 1.0e-20f, 3.0e-20f);
        f.zeroizeSubEpsilonRelativeInplace(1);
        assertEquals("float: a tiny real part beside a large imaginary one stays", 1.0e-20f,
                f.get(1, 0).re(), 0.0f);
        assertEquals("float: a tiny modulus falls", 0.0f, f.get(0, 1).re(), 0.0f);
        assertEquals("float: and its imaginary part too", 0.0f, f.get(0, 1).im(), 0.0f);
    }

    @Test
    public void testDegenerateMatricesAreLeftAlone() {
        MatrixD zero = Matrices.createD(3, 3);
        zero.zeroizeSubEpsilonRelativeInplace(1);
        assertEquals("the zero matrix stays zero", 0, nonZeroD(zero));

        MatrixD nan = Matrices.createD(3, 3);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                nan.set(i, j, Double.NaN);
            }
        }
        nan.zeroizeSubEpsilonRelativeInplace(1);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                assertTrue("an all NaN matrix is untouched", Double.isNaN(nan.get(i, j)));
            }
        }

        MatrixD one = Matrices.createD(1, 1);
        one.set(0, 0, 1.0e-300);
        one.zeroizeSubEpsilonRelativeInplace(1);
        assertEquals("a single entry is its own maximum", 1.0e-300, one.get(0, 0), 0.0);
    }

    @Test
    public void testTheMultiplierIsValidated() {
        for (int k : new int[] { 0, -1, Integer.MIN_VALUE }) {
            String at = " for k = " + k;
            rejects("MatrixD" + at, () -> Matrices.createD(2, 2).zeroizeSubEpsilonRelativeInplace(k));
            rejects("MatrixF" + at, () -> Matrices.createF(2, 2).zeroizeSubEpsilonRelativeInplace(k));
            rejects("ComplexMatrixD" + at,
                    () -> Matrices.createComplexD(2, 2).zeroizeSubEpsilonRelativeInplace(k));
            rejects("ComplexMatrixF" + at,
                    () -> Matrices.createComplexF(2, 2).zeroizeSubEpsilonRelativeInplace(k));
        }
        // k == 1 is the smallest legal multiplier
        assertEquals("k = 1 is accepted", N * N,
                nonZeroD(noiseless(1.0).zeroizeSubEpsilonRelativeInplace(1)));
    }

    @Test
    public void testALargerMultiplierRemovesMore() {
        MatrixD small = Matrices.createD(1, 3);
        small.set(0, 0, 1.0);
        small.set(0, 1, 1.0e-15);
        small.set(0, 2, 1.0e-13);
        MatrixD a = small.copy().zeroizeSubEpsilonRelativeInplace(1);
        assertEquals("k = 1 keeps 1e-15", 1.0e-15, a.get(0, 1), 0.0);
        MatrixD b = small.copy().zeroizeSubEpsilonRelativeInplace(100);
        assertEquals("k = 100 drops 1e-15", 0.0, b.get(0, 1), 0.0);
        assertEquals("k = 100 keeps 1e-13", 1.0e-13, b.get(0, 2), 0.0);
    }


    @Test
    public void testTheThresholdItselfIsRemovedButNothingAboveIt() {
        // with a maximum of 1.0 and k == 1 the threshold is exactly the epsilon
        MatrixD d = Matrices.createD(1, 3);
        d.set(0, 0, 1.0);
        d.set(0, 1, DimensionsBase.MACH_EPS_DBL);
        d.set(0, 2, Math.nextUp(DimensionsBase.MACH_EPS_DBL));
        d.zeroizeSubEpsilonRelativeInplace(1);
        assertEquals("the maximum stays", 1.0, d.get(0, 0), 0.0);
        assertEquals("an entry at the threshold goes", 0.0, d.get(0, 1), 0.0);
        assertEquals("the next value up stays", Math.nextUp(DimensionsBase.MACH_EPS_DBL),
                d.get(0, 2), 0.0);

        MatrixF f = Matrices.createF(1, 3);
        f.set(0, 0, 1.0f);
        f.set(0, 1, DimensionsBase.MACH_EPS_FLT);
        f.set(0, 2, Math.nextUp(DimensionsBase.MACH_EPS_FLT));
        f.zeroizeSubEpsilonRelativeInplace(1);
        assertEquals("float maximum stays", 1.0f, f.get(0, 0), 0.0f);
        assertEquals("float entry at the threshold goes", 0.0f, f.get(0, 1), 0.0f);
        assertEquals("float next value up stays", Math.nextUp(DimensionsBase.MACH_EPS_FLT),
                f.get(0, 2), 0.0f);

        ComplexMatrixD c = Matrices.createComplexD(1, 3);
        c.set(0, 0, 1.0, 0.0);
        c.set(0, 1, DimensionsBase.MACH_EPS_DBL, 0.0);
        c.set(0, 2, Math.nextUp(DimensionsBase.MACH_EPS_DBL), 0.0);
        c.zeroizeSubEpsilonRelativeInplace(1);
        assertEquals("complex maximum stays", 1.0, c.get(0, 0).re(), 0.0);
        assertEquals("complex entry at the threshold goes", 0.0, c.get(0, 1).re(), 0.0);
        assertEquals("complex next value up stays", Math.nextUp(DimensionsBase.MACH_EPS_DBL),
                c.get(0, 2).re(), 0.0);

        ComplexMatrixF cf = Matrices.createComplexF(1, 3);
        cf.set(0, 0, 1.0f, 0.0f);
        cf.set(0, 1, DimensionsBase.MACH_EPS_FLT, 0.0f);
        cf.set(0, 2, Math.nextUp(DimensionsBase.MACH_EPS_FLT), 0.0f);
        cf.zeroizeSubEpsilonRelativeInplace(1);
        assertEquals("complex float entry at the threshold goes", 0.0f, cf.get(0, 1).re(), 0.0f);
        assertEquals("complex float next value up stays", Math.nextUp(DimensionsBase.MACH_EPS_FLT),
                cf.get(0, 2).re(), 0.0f);
    }

    // ---- fixtures ----

    private static MatrixD noiseless(double scale) {
        return Matrices.randomUniformD(N, N, 0.5, 1.0, 5L).scaleInplace(scale);
    }

    private static MatrixD noisy(double scale) {
        MatrixD m = noiseless(scale);
        double[] a = m.getArrayUnsafe();
        for (int i = 0; i < a.length; i += NOISE_STRIDE) {
            a[i] = 1.0e-18 * scale;
        }
        return m;
    }

    private static MatrixF noisyF(double scale) {
        MatrixF m = Matrices.randomUniformF(N, N, 0.5f, 1.0f, 5L).scaleInplace((float) scale);
        float[] a = m.getArrayUnsafe();
        for (int i = 0; i < a.length; i += NOISE_STRIDE) {
            a[i] = (float) (1.0e-18 * scale);
        }
        return m;
    }

    private static ComplexMatrixD noisyComplex(double scale) {
        MatrixD r = noisy(scale);
        ComplexMatrixD m = Matrices.createComplexD(N, N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                m.set(i, j, r.get(i, j), 0.0);
            }
        }
        return m;
    }

    private static ComplexMatrixF noisyComplexF(double scale) {
        MatrixF r = noisyF(scale);
        ComplexMatrixF m = Matrices.createComplexF(N, N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                m.set(i, j, r.get(i, j), 0.0f);
            }
        }
        return m;
    }

    /** how many entries are not relative noise */
    private static int survivors(MatrixD m) {
        double[] a = m.getArrayUnsafe();
        int c = 0;
        for (int i = 0; i < a.length; ++i) {
            if (i % NOISE_STRIDE != 0) {
                ++c;
            }
        }
        return c;
    }

    // ---- counting ----

    private static int nonZeroD(MatrixD m) {
        int c = 0;
        for (double x : m.getArrayUnsafe()) {
            if (x != 0.0) {
                ++c;
            }
        }
        return c;
    }

    private static int nonZeroF(MatrixF m) {
        int c = 0;
        for (float x : m.getArrayUnsafe()) {
            if (x != 0.0f) {
                ++c;
            }
        }
        return c;
    }

    private static int nonZeroComplexD(ComplexMatrixD m) {
        double[] a = m.getArrayUnsafe();
        int c = 0;
        for (int i = 0; i < a.length; i += 2) {
            if (a[i] != 0.0 || a[i + 1] != 0.0) {
                ++c;
            }
        }
        return c;
    }

    private static int nonZeroComplexF(ComplexMatrixF m) {
        float[] a = m.getArrayUnsafe();
        int c = 0;
        for (int i = 0; i < a.length; i += 2) {
            if (a[i] != 0.0f || a[i + 1] != 0.0f) {
                ++c;
            }
        }
        return c;
    }

    private static double maxAbsD(MatrixD m) {
        double max = 0.0;
        for (double x : m.getArrayUnsafe()) {
            max = Math.max(max, Math.abs(x));
        }
        return max;
    }

    private static double maxAbsComplexD(ComplexMatrixD m) {
        double[] a = m.getArrayUnsafe();
        double max = 0.0;
        for (int i = 0; i < a.length; i += 2) {
            max = Math.max(max, Math.hypot(a[i], a[i + 1]));
        }
        return max;
    }

    private interface Body {
        void run();
    }

    private static void rejects(String what, Body body) {
        try {
            body.run();
            fail(what + " : expected an IllegalArgumentException but none was thrown");
        } catch (IllegalArgumentException expected) {
            // the guard did its job
        }
    }
}
