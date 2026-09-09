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

import org.junit.Test;

/**
 * Elementwise {@code abs()} on the four matrix types, bit for bit, including
 * the signed zeros.
 */
public final class MatrixAbsTest {

    private static final double INF = Double.POSITIVE_INFINITY;
    private static final double NAN = Double.NaN;

    private static void same(String what, double want, double got) {
        assertEquals(what + ": want " + want + ", got " + got, Double.doubleToLongBits(want),
                Double.doubleToLongBits(got));
    }

    private static void same(String what, float want, float got) {
        assertEquals(what + ": want " + want + ", got " + got, Float.floatToIntBits(want),
                Float.floatToIntBits(got));
    }

    private static double[] absD(double re, double im) {
        ComplexMatrixD m = Matrices.createComplexD(1, 1);
        m.set(0, 0, re, im);
        return m.abs().getArrayUnsafe();
    }

    private static float[] absF(float re, float im) {
        ComplexMatrixF m = Matrices.createComplexF(1, 1);
        m.set(0, 0, re, im);
        return m.abs().getArrayUnsafe();
    }

    private static double absReal(double x) {
        MatrixD m = Matrices.createD(1, 1);
        m.set(0, 0, x);
        return m.abs().getArrayUnsafe()[0];
    }

    private static float absReal(float x) {
        MatrixF m = Matrices.createF(1, 1);
        m.set(0, 0, x);
        return m.abs().getArrayUnsafe()[0];
    }

    @Test
    public void testRealMatrixAbs() {
        same("abs(3)", 3.0, absReal(3.0));
        same("abs(-3)", 3.0, absReal(-3.0));
        same("abs(0.0)", 0.0, absReal(0.0));
        // a modulus carries no sign
        same("abs(-0.0)", 0.0, absReal(-0.0));
        same("abs(-inf)", INF, absReal(-INF));
        same("abs(NaN)", NAN, absReal(NAN));
    }

    @Test
    public void testRealMatrixAbsInSinglePrecision() {
        same("abs(3)", 3.0f, absReal(3.0f));
        same("abs(-3)", 3.0f, absReal(-3.0f));
        same("abs(-0.0)", 0.0f, absReal(-0.0f));
        same("abs(-inf)", (float) INF, absReal((float) -INF));
    }

    @Test
    public void testComplexMatrixAbsOnTheRealAxis() {
        same("abs(3+0i) re", 3.0, absD(3.0, 0.0)[0]);
        same("abs(3+0i) im", 0.0, absD(3.0, 0.0)[1]);
        same("abs(-3+0i) re", 3.0, absD(-3.0, 0.0)[0]);
        // a modulus carries no sign, and the result is real
        same("abs(-0.0+0i) re", 0.0, absD(-0.0, 0.0)[0]);
        same("abs(-0.0+0i) im", 0.0, absD(-0.0, 0.0)[1]);
        same("abs(-0.0-0.0i) re", 0.0, absD(-0.0, -0.0)[0]);
        same("abs(-0.0-0.0i) im", 0.0, absD(-0.0, -0.0)[1]);
        same("abs(3-0.0i) im", 0.0, absD(3.0, -0.0)[1]);
    }

    @Test
    public void testComplexMatrixAbsOffTheRealAxis() {
        same("abs(3+4i) re", 5.0, absD(3.0, 4.0)[0]);
        same("abs(3+4i) im", 0.0, absD(3.0, 4.0)[1]);
        same("abs(-3-4i) re", 5.0, absD(-3.0, -4.0)[0]);
        same("abs(-3-4i) im", 0.0, absD(-3.0, -4.0)[1]);
    }

    @Test
    public void testComplexMatrixAbsInSinglePrecision() {
        same("abs(3+4i) re", 5.0f, absF(3.0f, 4.0f)[0]);
        same("abs(3+4i) im", 0.0f, absF(3.0f, 4.0f)[1]);
        same("abs(-0.0+0i) re", 0.0f, absF(-0.0f, 0.0f)[0]);
        same("abs(-0.0-0.0i) re", 0.0f, absF(-0.0f, -0.0f)[0]);
        same("abs(-0.0-0.0i) im", 0.0f, absF(-0.0f, -0.0f)[1]);
        same("abs(3-0.0i) im", 0.0f, absF(3.0f, -0.0f)[1]);
    }

    @Test
    public void testComplexMatrixAbsLeavesTheRestAlone() {
        // every element, not just the first
        ComplexMatrixD m = Matrices.createComplexD(2, 2);
        m.set(0, 0, -0.0, 0.0);
        m.set(0, 1, -3.0, 4.0);
        m.set(1, 0, 2.0, -0.0);
        m.set(1, 1, -1.0, 0.0);
        ComplexMatrixD abs = m.abs();
        double[] a = abs.getArrayUnsafe();
        for (int i = 0; i < a.length; i += 2) {
            // the sign bit is clear for every nonnegative value, -0.0 included
            assertTrue("element " + (i / 2) + " carries a sign, " + a[i], Double.doubleToLongBits(a[i]) >= 0L);
            same("element " + (i / 2) + " im", 0.0, a[i + 1]);
        }
        same("|(-3,4)| re", 5.0, abs.get(0, 1).re());
        same("|(2,-0.0)| re", 2.0, abs.get(1, 0).re());
        same("|(-1,0)| re", 1.0, abs.get(1, 1).re());
    }
}
