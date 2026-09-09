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
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.fail;

import org.junit.Test;

/**
 * Pins the six broadcasted column vector operations of {@code MatrixD} and
 * {@code MatrixF}: {@code addBroadcastedVectorInplace} /
 * {@code plusBroadcastedVector}, {@code mulBroadcastedVectorInplace} /
 * {@code mulBroadcastedVector} and {@code divBroadcastedVectorInplace} /
 * {@code divBroadcastedVector}.
 * <p>
 * A column vector must act as if it had been stretched across all columns, a
 * matrix of equal dimension must act elementwise, and any other dimension must
 * be rejected. The expected values come from a scalar loop written out in this
 * class, in the same operand order as the implementation, so the comparison is
 * on raw bits. Nothing here touches BLAS or LAPACK.
 */
public final class BroadcastedVectorTest {

    private static final long SEED = 20260909L;

    /** the three operations, in the order the reference loop switches on */
    private static final char ADD = '+';
    private static final char MUL = '*';
    private static final char DIV = '/';

    private static final int[][] SHAPES = { { 1, 1 }, { 1, 5 }, { 5, 1 }, { 4, 3 }, { 3, 4 }, { 40, 7 },
            { 7, 40 } };

    // ---------------------------------------------------------------- tests

    @Test
    public void testAColumnVectorIsStretchedAcrossAllColumns() {
        for (int[] s : SHAPES) {
            String at = " at " + s[0] + "x" + s[1];
            for (char op : new char[] { ADD, MUL, DIV }) {
                MatrixD d = Matrices.randomUniformD(s[0], s[1], 1.0, 2.0, SEED);
                MatrixD vd = Matrices.randomUniformD(s[0], 1, 1.0, 2.0, SEED + 1L);
                double[] wantD = refD(d.getArrayUnsafe(), vd.getArrayUnsafe(), s[0], s[1], true, op);
                assertSame(op + " MatrixD returns this" + at, d, applyD(d, vd, op));
                assertBitsD(op + " MatrixD" + at, wantD, d.getArrayUnsafe());

                MatrixF f = Matrices.randomUniformF(s[0], s[1], 1.0f, 2.0f, SEED);
                MatrixF vf = Matrices.randomUniformF(s[0], 1, 1.0f, 2.0f, SEED + 1L);
                float[] wantF = refF(f.getArrayUnsafe(), vf.getArrayUnsafe(), s[0], s[1], true, op);
                assertSame(op + " MatrixF returns this" + at, f, applyF(f, vf, op));
                assertBitsF(op + " MatrixF" + at, wantF, f.getArrayUnsafe());
            }
        }
    }

    @Test
    public void testAMatrixOfEqualDimensionActsElementwise() {
        for (int[] s : SHAPES) {
            String at = " at " + s[0] + "x" + s[1];
            for (char op : new char[] { ADD, MUL, DIV }) {
                MatrixD d = Matrices.randomUniformD(s[0], s[1], 1.0, 2.0, SEED);
                MatrixD bd = Matrices.randomUniformD(s[0], s[1], 1.0, 2.0, SEED + 1L);
                double[] wantD = refD(d.getArrayUnsafe(), bd.getArrayUnsafe(), s[0], s[1], false, op);
                applyD(d, bd, op);
                assertBitsD(op + " MatrixD" + at, wantD, d.getArrayUnsafe());

                MatrixF f = Matrices.randomUniformF(s[0], s[1], 1.0f, 2.0f, SEED);
                MatrixF bf = Matrices.randomUniformF(s[0], s[1], 1.0f, 2.0f, SEED + 1L);
                float[] wantF = refF(f.getArrayUnsafe(), bf.getArrayUnsafe(), s[0], s[1], false, op);
                applyF(f, bf, op);
                assertBitsF(op + " MatrixF" + at, wantF, f.getArrayUnsafe());
            }
        }
    }

    @Test
    public void testTheOneColumnCaseAgreesWithTheEqualDimensionCase() {
        // a 1 column matrix hits the equal dimension branch, a wider one the
        // broadcast branch; stretching by hand must give the same answer
        for (int[] s : new int[][] { { 6, 4 }, { 40, 3 }, { 3, 40 } }) {
            for (char op : new char[] { ADD, MUL, DIV }) {
                MatrixD d = Matrices.randomUniformD(s[0], s[1], 1.0, 2.0, SEED);
                MatrixD v = Matrices.randomUniformD(s[0], 1, 1.0, 2.0, SEED + 1L);
                MatrixD stretched = Matrices.createD(s[0], s[1]);
                for (int col = 0; col < s[1]; ++col) {
                    stretched.setSubmatrixInplace(0, col, v, 0, 0, s[0] - 1, 0);
                }
                MatrixD viaVector = applyD(d.copy(), v, op);
                MatrixD viaMatrix = applyD(d.copy(), stretched, op);
                assertBitsD(op + " broadcast vs stretched at " + s[0] + "x" + s[1],
                        viaMatrix.getArrayUnsafe(), viaVector.getArrayUnsafe());
            }
        }
    }

    @Test
    public void testTheCopyVariantsLeaveTheReceiverAlone() {
        for (int[] s : SHAPES) {
            String at = " at " + s[0] + "x" + s[1];
            MatrixD d = Matrices.randomUniformD(s[0], s[1], 1.0, 2.0, SEED);
            MatrixD vd = Matrices.randomUniformD(s[0], 1, 1.0, 2.0, SEED + 1L);
            double[] untouchedD = d.getArrayUnsafe().clone();
            MatrixD pd = d.plusBroadcastedVector(vd);
            MatrixD md = d.mulBroadcastedVector(vd);
            MatrixD qd = d.divBroadcastedVector(vd);
            assertBitsD("MatrixD receiver untouched" + at, untouchedD, d.getArrayUnsafe());
            assertNotSame("plusBroadcastedVector is a copy" + at, d, pd);
            assertNotSame("mulBroadcastedVector is a copy" + at, d, md);
            assertNotSame("divBroadcastedVector is a copy" + at, d, qd);
            assertBitsD("plusBroadcastedVector" + at, refD(untouchedD, vd.getArrayUnsafe(), s[0], s[1], true,
                    ADD), pd.getArrayUnsafe());
            assertBitsD("mulBroadcastedVector" + at, refD(untouchedD, vd.getArrayUnsafe(), s[0], s[1], true,
                    MUL), md.getArrayUnsafe());
            assertBitsD("divBroadcastedVector" + at, refD(untouchedD, vd.getArrayUnsafe(), s[0], s[1], true,
                    DIV), qd.getArrayUnsafe());

            MatrixF f = Matrices.randomUniformF(s[0], s[1], 1.0f, 2.0f, SEED);
            MatrixF vf = Matrices.randomUniformF(s[0], 1, 1.0f, 2.0f, SEED + 1L);
            float[] untouchedF = f.getArrayUnsafe().clone();
            MatrixF pf = f.plusBroadcastedVector(vf);
            MatrixF mf = f.mulBroadcastedVector(vf);
            MatrixF qf = f.divBroadcastedVector(vf);
            assertBitsF("MatrixF receiver untouched" + at, untouchedF, f.getArrayUnsafe());
            assertNotSame("plusBroadcastedVector is a copy" + at, f, pf);
            assertNotSame("mulBroadcastedVector is a copy" + at, f, mf);
            assertNotSame("divBroadcastedVector is a copy" + at, f, qf);
            assertBitsF("plusBroadcastedVector" + at, refF(untouchedF, vf.getArrayUnsafe(), s[0], s[1], true,
                    ADD), pf.getArrayUnsafe());
            assertBitsF("mulBroadcastedVector" + at, refF(untouchedF, vf.getArrayUnsafe(), s[0], s[1], true,
                    MUL), mf.getArrayUnsafe());
            assertBitsF("divBroadcastedVector" + at, refF(untouchedF, vf.getArrayUnsafe(), s[0], s[1], true,
                    DIV), qf.getArrayUnsafe());
        }
    }

    @Test
    public void testDivisionByZeroFollowsIeee754() {
        MatrixD d = Matrices.createD(4, 2);
        double[] numerator = { 1.0, -1.0, 0.0, Double.NaN };
        for (int col = 0; col < 2; ++col) {
            for (int row = 0; row < 4; ++row) {
                d.set(row, col, numerator[row]);
            }
        }
        MatrixD v = Matrices.createD(4, 1);
        for (int row = 0; row < 4; ++row) {
            v.set(row, 0, 0.0);
        }
        d.divBroadcastedVectorInplace(v);
        // 1/0 = +Inf, -1/0 = -Inf, 0/0 = NaN, NaN/0 = NaN, in both columns
        double[] want = { Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NaN, Double.NaN,
                Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NaN, Double.NaN };
        assertBitsD("MatrixD division by zero", want, d.getArrayUnsafe());

        MatrixD negZero = Matrices.createD(4, 1);
        for (int row = 0; row < 4; ++row) {
            negZero.set(row, 0, -0.0);
        }
        MatrixD e = Matrices.createD(4, 1);
        for (int row = 0; row < 4; ++row) {
            e.set(row, 0, numerator[row]);
        }
        e.divBroadcastedVectorInplace(negZero);
        double[] wantNeg = { Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, Double.NaN, Double.NaN };
        assertBitsD("MatrixD division by negative zero", wantNeg, e.getArrayUnsafe());
    }

    @Test
    public void testMultiplicationKeepsTheSignOfZero() {
        MatrixD d = Matrices.createD(2, 2);
        d.set(0, 0, 0.0);
        d.set(1, 0, -0.0);
        d.set(0, 1, 0.0);
        d.set(1, 1, -0.0);
        MatrixD v = Matrices.createD(2, 1);
        v.set(0, 0, -1.0);
        v.set(1, 0, -1.0);
        d.mulBroadcastedVectorInplace(v);
        double[] want = { -0.0, 0.0, -0.0, 0.0 };
        assertBitsD("MatrixD signed zero under multiplication", want, d.getArrayUnsafe());
    }

    @Test
    public void testMismatchedDimensionsAreRejected() {
        MatrixD a = Matrices.randomUniformD(4, 3, SEED);
        MatrixD wrongRows = Matrices.randomUniformD(5, 1, SEED);
        MatrixD wrongCols = Matrices.randomUniformD(4, 2, SEED);
        rejects("mul: wrong rows", () -> a.mulBroadcastedVectorInplace(wrongRows));
        rejects("div: wrong rows", () -> a.divBroadcastedVectorInplace(wrongRows));
        rejects("add: wrong rows", () -> a.addBroadcastedVectorInplace(wrongRows));
        rejects("mul: neither equal nor a column vector", () -> a.mulBroadcastedVectorInplace(wrongCols));
        rejects("div: neither equal nor a column vector", () -> a.divBroadcastedVectorInplace(wrongCols));
        rejects("add: neither equal nor a column vector", () -> a.addBroadcastedVectorInplace(wrongCols));
        rejects("mul copy: wrong rows", () -> a.mulBroadcastedVector(wrongRows));
        rejects("div copy: wrong rows", () -> a.divBroadcastedVector(wrongRows));
        rejects("mul copy: neither equal nor a column vector", () -> a.mulBroadcastedVector(wrongCols));
        rejects("div copy: neither equal nor a column vector", () -> a.divBroadcastedVector(wrongCols));

        MatrixF b = Matrices.randomUniformF(4, 3, SEED);
        MatrixF wrongRowsF = Matrices.randomUniformF(5, 1, SEED);
        MatrixF wrongColsF = Matrices.randomUniformF(4, 2, SEED);
        rejects("MatrixF mul: wrong rows", () -> b.mulBroadcastedVectorInplace(wrongRowsF));
        rejects("MatrixF div: wrong rows", () -> b.divBroadcastedVectorInplace(wrongRowsF));
        rejects("MatrixF mul: bad column count", () -> b.mulBroadcastedVectorInplace(wrongColsF));
        rejects("MatrixF div: bad column count", () -> b.divBroadcastedVectorInplace(wrongColsF));
        rejects("MatrixF mul copy: bad column count", () -> b.mulBroadcastedVector(wrongColsF));
        rejects("MatrixF div copy: bad column count", () -> b.divBroadcastedVector(wrongColsF));
    }

    @Test
    public void testARejectedCallLeavesTheMatrixUntouched() {
        MatrixD a = Matrices.randomUniformD(4, 3, 1.0, 2.0, SEED);
        double[] untouched = a.getArrayUnsafe().clone();
        MatrixD wrongCols = Matrices.randomUniformD(4, 2, 1.0, 2.0, SEED);
        rejects("mul", () -> a.mulBroadcastedVectorInplace(wrongCols));
        assertBitsD("after a rejected mul", untouched, a.getArrayUnsafe());
        rejects("div", () -> a.divBroadcastedVectorInplace(wrongCols));
        assertBitsD("after a rejected div", untouched, a.getArrayUnsafe());
    }

    // ------------------------------------------------------------- helpers

    private static MatrixD applyD(MatrixD A, MatrixD B, char op) {
        switch (op) {
        case ADD:
            return A.addBroadcastedVectorInplace(B);
        case MUL:
            return A.mulBroadcastedVectorInplace(B);
        default:
            return A.divBroadcastedVectorInplace(B);
        }
    }

    private static MatrixF applyF(MatrixF A, MatrixF B, char op) {
        switch (op) {
        case ADD:
            return A.addBroadcastedVectorInplace(B);
        case MUL:
            return A.mulBroadcastedVectorInplace(B);
        default:
            return A.divBroadcastedVectorInplace(B);
        }
    }

    /**
     * The expected result, computed the slow and obvious way. {@code broadcast}
     * selects between reading {@code b} as a column vector (one value per row)
     * and reading it as a full matrix.
     */
    private static double[] refD(double[] a, double[] b, int rows, int cols, boolean broadcast, char op) {
        double[] out = a.clone();
        for (int col = 0; col < cols; ++col) {
            for (int row = 0; row < rows; ++row) {
                int i = col * rows + row;
                double y = broadcast ? b[row] : b[i];
                if (op == ADD) {
                    out[i] = out[i] + y;
                } else if (op == MUL) {
                    out[i] = out[i] * y;
                } else {
                    out[i] = out[i] / y;
                }
            }
        }
        return out;
    }

    private static float[] refF(float[] a, float[] b, int rows, int cols, boolean broadcast, char op) {
        float[] out = a.clone();
        for (int col = 0; col < cols; ++col) {
            for (int row = 0; row < rows; ++row) {
                int i = col * rows + row;
                float y = broadcast ? b[row] : b[i];
                if (op == ADD) {
                    out[i] = out[i] + y;
                } else if (op == MUL) {
                    out[i] = out[i] * y;
                } else {
                    out[i] = out[i] / y;
                }
            }
        }
        return out;
    }

    /**
     * Bitwise comparison, but with all NaNs collapsed onto one value: these
     * operations compute, and the bit pattern of an arithmetically produced NaN
     * is not specified (on x86 {@code 0.0 / 0.0} sets the sign bit, the
     * {@code Double.NaN} constant does not). The signed zeros stay
     * distinguishable, which is the point of comparing bits at all here.
     */
    private static void assertBitsD(String what, double[] expected, double[] actual) {
        assertEquals(what + " length", expected.length, actual.length);
        for (int i = 0; i < expected.length; ++i) {
            long e = Double.doubleToLongBits(expected[i]);
            long a = Double.doubleToLongBits(actual[i]);
            if (e != a) {
                fail(what + " differs at " + i + " : expected " + expected[i] + " but was " + actual[i]);
            }
        }
    }

    private static void assertBitsF(String what, float[] expected, float[] actual) {
        assertEquals(what + " length", expected.length, actual.length);
        for (int i = 0; i < expected.length; ++i) {
            int e = Float.floatToIntBits(expected[i]);
            int a = Float.floatToIntBits(actual[i]);
            if (e != a) {
                fail(what + " differs at " + i + " : expected " + expected[i] + " but was " + actual[i]);
            }
        }
    }

    private interface Body {
        void run();
    }

    private static void rejects(String what, Body body) {
        try {
            body.run();
            fail(what + " : expected an IndexOutOfBoundsException but none was thrown");
        } catch (IndexOutOfBoundsException expected) {
            // the guard did its job
        }
    }
}
