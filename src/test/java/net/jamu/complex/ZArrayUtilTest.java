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
package net.jamu.complex;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Test;

import net.jamu.matrix.ComplexMatrixD;
import net.jamu.matrix.ComplexMatrixF;
import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixD;

/**
 * Tests for the L2 norm, which is what ComplexMatrix.normF() is built on.
 */
public final class ZArrayUtilTest {

    private static final int ROWS = 40;
    private static final int COLS = 30;
    /** norm of a ROWS x COLS matrix whose entries are all (1, 0) */
    private static final double UNIT_NORM = Math.sqrt(ROWS * (double) COLS);

    /** the scales a double still represents as a normal number */
    private static final double[] SCALES_D = { 1.0e+300, 1.0e+150, 1.0, 1.0e-150, 1.0e-159, 1.0e-200, 1.0e-300 };
    /** the same for single precision */
    private static final float[] SCALES_F = { 1.0e+30f, 1.0f, 1.0e-10f, 1.0e-20f, 1.0e-30f, 1.0e-38f };

    private static void assertClose(String what, double expected, double actual, double tol) {
        assertTrue(what + ": expected " + expected + " but was " + actual,
                Math.abs(expected - actual) <= tol * Math.abs(expected));
    }

    private static ComplexMatrixD filledD(double re, double im) {
        ComplexMatrixD A = Matrices.createComplexD(ROWS, COLS);
        for (int j = 0; j < COLS; ++j) {
            for (int i = 0; i < ROWS; ++i) {
                A.set(i, j, re, im);
            }
        }
        return A;
    }

    private static ComplexMatrixF filledF(float re, float im) {
        ComplexMatrixF A = Matrices.createComplexF(ROWS, COLS);
        for (int j = 0; j < COLS; ++j) {
            for (int i = 0; i < ROWS; ++i) {
                A.set(i, j, re, im);
            }
        }
        return A;
    }

    private static MatrixD filledReal(double x) {
        MatrixD A = Matrices.createD(ROWS, COLS);
        for (int j = 0; j < COLS; ++j) {
            for (int i = 0; i < ROWS; ++i) {
                A.set(i, j, x);
            }
        }
        return A;
    }

    @Test
    public void testScaleInvarianceDouble() {
        for (double s : SCALES_D) {
            assertClose("scale " + s, 5.0 * s, ZArrayUtil.l2norm(new double[] { 3.0 * s, 4.0 * s }), 1.0e-15);
        }
    }

    @Test
    public void testSubnormalDouble() {
        // precision is reduced down here, the point is that it is not 0
        assertClose("subnormal", 5.0e-320, ZArrayUtil.l2norm(new double[] { 3.0e-320, 4.0e-320 }), 1.0e-10);
    }

    @Test
    public void testScaleInvarianceFloat() {
        for (float s : SCALES_F) {
            assertClose("scale " + s, 5.0f * s, ZArrayUtil.l2norm(new float[] { 3.0f * s, 4.0f * s }), 1.0e-6);
        }
    }

    @Test
    public void testComplexNormFDouble() {
        for (double s : SCALES_D) {
            assertClose("scale " + s, s * UNIT_NORM, filledD(s, 0.0).normF(), 1.0e-15);
        }
    }

    @Test
    public void testComplexNormFFloat() {
        for (float s : SCALES_F) {
            assertClose("scale " + s, s * UNIT_NORM, filledF(s, 0.0f).normF(), 1.0e-6);
        }
    }

    @Test
    public void testComplexAgreesWithReal() {
        for (double s : SCALES_D) {
            assertEquals("scale " + s, filledReal(s).normF(), filledD(s, 0.0).normF(), 0.0);
        }
    }

    @Test
    public void testLargeComponentsDoNotOverflow() {
        // |re| + |im| overflows here, the norm itself does not
        assertClose("large", Math.sqrt(2.0) * 1.0e+308, ZArrayUtil.l2norm(new double[] { 1.0e+308, 1.0e+308 }),
                1.0e-15);
    }

    @Test
    public void testEdgeCases() {
        assertEquals(0.0, ZArrayUtil.l2norm((double[]) null), 0.0);
        assertEquals(0.0, ZArrayUtil.l2norm(new double[] {}), 0.0);
        assertEquals(0.0, ZArrayUtil.l2norm(new double[] { 0.0, 0.0, 0.0, 0.0 }), 0.0);
        assertEquals(0.0f, ZArrayUtil.l2norm((float[]) null), 0.0f);
        assertEquals(0.0f, ZArrayUtil.l2norm(new float[] { 0.0f, 0.0f }), 0.0f);
        try {
            ZArrayUtil.l2norm(new double[] { 1.0, 2.0, 3.0 });
            fail("expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
            // as documented
        }
    }
}
